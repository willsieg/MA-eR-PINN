import torch
import bisect
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
SEED = 42; random.seed(SEED); torch.manual_seed(SEED)


###################################################################################################################################
# -----------------------------------------------------------------------------------------------------------
# DATASET DEFINITION (SEQUENCE LEVEL) -----------------------------------------------------------------------
class TripDataset(Dataset):
    def __init__(self, file_list: list, input_columns: list, target_column: str, prior_column: str, scaler, target_scaler, prior_scaler, fit: bool = False):
        self.file_list = file_list
        self.input_columns = input_columns
        self.target_column = target_column
        self.prior_column = prior_columncolumn
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.prior_scaler = prior_scaler
        self.fit = fit
        self.data = []
        self.targets = []
        self.priors = []
        self.cumulative_lengths = []

        # fitting scalers over complete training dataset
        if self.fit:
            print(f"fitting Scalers: {scaler.__class__.__name__}, {target_scaler.__class__.__name__}")
            # Initialize and Fit the scalers on the complete training data set
            # Fit the scalers incrementally to avoid memory errors
            num_files = len(self.file_list)
            for i, file in enumerate(self.file_list):
                df = pd.read_parquet(file, columns = input_columns+target_column+prior_column, engine='pyarrow')
                X = df[input_columns].values
                y = df[target_column].values.reshape(-1, 1)  # Reshape to match the shape of the input: 2D array with one column
                P = df[prior_column].values.reshape(-1, 1)
                self.scaler.partial_fit(X)
                self.target_scaler.partial_fit(y)
                self.prior_scaler.partial_fit(P)
            
                # Print status info at 50%
                if i == num_files // 2: print(f"\t50% of the fitting done...")
                
            print(f"Done. Create DataSets and DataLoaders...")

        # transform with fitted scalers
        cumulative_length = 0
        for i, file in enumerate(self.file_list):
            # DATA PREPROCESSING -----------------------------------------------------------
            # Assigning inputs and targets and reshaping ---------------
            df = pd.read_parquet(file, columns = input_columns+target_column, engine='pyarrow')
            X = df[input_columns].values
            y = df[target_column].values.reshape(-1, 1)  # Reshape 
            # use the previously fitted scalers to transform the data

            X = self.scaler.transform(X)  
            y = self.target_scaler.transform(y).squeeze()  # is .squeeze() necessary here?
            # Append to data
            self.data.append(torch.tensor(X, dtype=torch.float32))
            self.targets.append(torch.tensor(y, dtype=torch.float32))
            cumulative_length += len(y)
            self.cumulative_lengths.append(cumulative_length)

        self.length = cumulative_length

    def __len__(self): return self.length

    def __getitem__(self, index):
        # Check if the index is within the valid range
        if index < 0 or index >= self.length:
            raise IndexError("Index out of range")
        
        # Use binary search to find the correct file and index
        file_idx = bisect.bisect_right(self.cumulative_lengths, index)
        if file_idx > 0:
            index -= self.cumulative_lengths[file_idx - 1]
        return (
            self.data[file_idx][index].unsqueeze(0),  # Add time dimension
            self.targets[file_idx][index]
        )


###################################################################################################################################
# -----------------------------------------------------------------------------------------------------------
# DATASET DEFINITION (BATCH LEVEL) -----------------------------------------------------------------------
class BatchDataset(Dataset):
    def __init__(self, batches: list):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int):
        return self.batches[idx]


# -----------------------------------------------------------------------------------------------------------
def create_batches(dataset: Dataset, batch_size: int, shuffle_batches: bool = True) -> BatchDataset:
    # Create a list to store the batches
    batches = []

    # Iterate through the sorted dataset in chunks of batch_size
    for i in range(0, len(dataset.targets), batch_size):
        batch = [(dataset.data[j], dataset.targets[j]) for j in range(i, min(i + batch_size, len(dataset.targets)))]
        batches.append(batch)

    # Shuffle the order of batches
    if shuffle_batches:
        random.shuffle(batches)

    # Ensure the shortest batch is placed at the end 
    batch_sizes = [len(batch) for batch in batches]
    # This is important in order to keep hidden/cell states of the LSTM across all batches
    shortest_batch = min(batches, key=len)
    batches.remove(shortest_batch)
    if len(shortest_batch) > 1: 
        batches.append(shortest_batch)
    else:
        print(f"\t --> Warning: Removed the shortest batch with size 1")
    
    '''
    # Discard batches with size 1 to avoid errors in the collate_fn
    if any(size == 1 for size in batch_sizes):
        batches = [batch for batch in batches if len(batch) > 1]
        print(f"\t --> Warning: Discarded {len(batch_sizes) - len(batches)} more batches with size 1")
    '''

    # Print the number of batches created
    print(f"\tNumber of batches created: {len(batches)}")

    # Print the sizes of the batches
    #print(f"\t\tBatch sizes: {batch_sizes}")
    
    return BatchDataset(batches)


# -----------------------------------------------------------------------------------------------------------
def collate_fn(batch, shuffle_in_batch: bool = False, padding_value: int = 0) -> tuple:
    """
    Custom collate function to process and pad sequences in a batch.
    Args:
        batch (list of tuples): A list of tuples where each tuple contains two elements:
            - inputs (torch.Tensor): The input sequence tensor.
            - targets (torch.Tensor): The target sequence tensor.
    Returns:
        tuple: A tuple containing:
            - packed_inputs (torch.nn.utils.rnn.PackedSequence): The packed input sequences.
            - padded_targets (torch.Tensor): The padded target sequences.
            - lengths (list of int): The original lengths of the input sequences.
    """

    # If batch is a list of several batches, unwrap and concatenate them
    if all(isinstance(b, list) for b in batch):
        batch = [item for sublist in batch for item in sublist]
        #print(f"batches unwrapped: {len(batch)}")

    # Check if each element of batch is a tuple containing two elements
    for n, element in enumerate(batch):
        if not isinstance(element, tuple) or len(element) != 2:
            raise ValueError("Each element of batch should be a tuple containing two elements")

    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=padding_value)

    # Shuffle the inputs, targets, and lengths inside the batch
    if shuffle_in_batch:
        indices = torch.randperm(padded_inputs.size(0))
        padded_inputs = padded_inputs[indices]
        padded_targets = padded_targets[indices]
        lengths = lengths[indices].tolist()

    # Pack the padded input sequences
    packed_inputs = pack_padded_sequence(padded_inputs, lengths, batch_first=True, \
        enforce_sorted = not shuffle_in_batch)

    return packed_inputs, padded_targets, lengths


# BATCH LOADER CHECK -----------------------------------------------------------------------
def check_batch(train_loader):
    # Iterate through the train_loader once and print a batch example of a PackedSequence
    for batch_idx, (packed_inputs, padded_targets, lengths) in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        print(f"Shape of packed_inputs.data: {packed_inputs.data.shape}")
        print(f"Lengths: {lengths}")
        # check correct types and shapes
        assert type(packed_inputs) == torch.nn.utils.rnn.PackedSequence
        assert type(packed_inputs.data) == torch.Tensor
        assert type(packed_inputs.batch_sizes) == torch.Tensor
        assert type(padded_targets) == torch.Tensor
        assert type(lengths) == torch.Tensor
        assert len(packed_inputs.batch_sizes) == max(lengths)
        assert sum(lengths) == packed_inputs.data.shape[0]
        break


###################################################################################################################################
# -----------------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS -----------------------------------------------------------------------------------------


# GENERATE DATALOADERS  ---------------------------------------------------------------------------------------
def prepare_dataloader(subset, indices_by_length, batch_size, input_columns, target_column, scaler, target_scaler, \
    dataloader_settings, fit=False, drop_last = False) -> tuple:
    """
    Prepares a DataLoader for the given dataset subset.

    Args:
        subset (Dataset): The dataset subset to be used.
        indices_by_length (list): List of indices sorted by sequence length.
        batch_size (int): The size of each batch.
        input_columns (list): List of column names to be used as input features.
        target_column (str): The column name to be used as the target variable.
        scaler (object): Scaler object for normalizing input features.
        target_scaler (object): Scaler object for normalizing target variable.
        dataloader_settings (dict): Additional settings for the DataLoader.
        fit (bool, optional): Whether to fit the scalers on the dataset. Defaults to False.

    Returns:
        tuple: A tuple containing the dataset, dataset batches, and DataLoader.

    DESCRIPTION ----------------------------------------------------------------
    Notes: for each of the three subsets, the following steps are performed:
    1. Sort each subset by descending sequence lengths based on the obtained indices
    2. Check if the number of samples leaves a remainder, leading to a (last) batch containing fewer samples. 
        In order to avoid later issues with tensor dimensions, this last (and therefore shortest batch) will be removed.
    3. Create a (custom) TripDataset object to select the input and target columns and apply the scalers. In case
         of the training subset, the scalers will be fitted to the training set first.
    4. Create a (custom) BatchDataset object of the corresponding TripDataset to handle the sequence padding before using 
            the DataLoader to create the batches.
    5. The DataLoader will then be used to iterate over the batches during training. To use the integrated collate_fn function
            of the DataLoader, the batch_size has to be set to 1. The actual batch size is then handled by the BatchDataset object.
    6. The collate_fn that is integrated in the DataLoader will automatically handle the shuffling, padding and packing
            of the sequences. The DataLoader will return a tuple of (packed_inputs, padded_targets, lengths), where
            the packed_inputs are PackedSequence objects that can be efficiently processed by RNNs.
            [Output tuple of types (<class 'torch.nn.utils.rnn.PackedSequence'>, <class 'torch.Tensor'>, <class 'torch.Tensor'>)]

    Note: shuffling will be done batchwise, however inside each batch the sequences will remain sorted by length

    *Note: Because of the BatchDataset object in the train loader, "batch_size" refers to the number of batches to feed, not the 
    number of samples in a batch. Also, the "drop_last" argument is useless due to this.
    """
    
    #subset_indices = [i for i in indices_by_length if i in set(list(subset.indices))]
    #if len(subset_indices) % batch_size == 1:
    #    subset_indices = subset_indices[:-1]
    #subset = torch.utils.data.Subset(subset, subset_indices)
    #dataset = TripDataset([subset.dataset.file_list[i] for i in subset.indices], input_columns, target_column, scaler, target_scaler, fit=fit)


    subset.indices = [i for i in indices_by_length if i in set(list(subset.indices))]

    remainder = len(subset) % batch_size
    if remainder != 0 and drop_last == True:
        subset.indices = subset.indices[:-(remainder)]
        print(f" --> Warning: Removed the last {remainder} samples to ensure a balanced batch size")

    dataset = TripDataset(subset, input_columns, target_column, scaler, target_scaler, fit=fit)
    
    
    dataset_batches = create_batches(dataset, batch_size)
    loader = DataLoader(dataset_batches, **dataloader_settings)
    return subset, dataset, dataset_batches, loader


###################################################################################################################################
# PLOT SEQUENCE PADDING PROCESS -----------------------------------------------------------------------------
def plot_padded_sequences(batch_size, trip_lengths, descr) -> float:
        """
        Plots the sequence lengths of batches with padding highlighted and calculates the ratio of padding to total length for each batch.
        Args:
            batch_size (int): The size of each batch.
            trip_lengths (list of int): A list containing the lengths of each sequence.
            descr (str): Description of the data loader (e.g. "Training", "Validation", or "Test").
        Returns:
            list of float: A list of ratios of padding to total length for each batch.
        """
        # Calculate the number of batches
        num_batches = int(np.ceil(len(trip_lengths) / batch_size))
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(18, 5))
        ratios = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(trip_lengths)) 
            batch_lengths = trip_lengths[start_idx:end_idx]

            ax.bar(range(start_idx, end_idx), batch_lengths, color='blue', edgecolor='blue') # Plot the actual sequence lengths

            max_length = max(batch_lengths)
            blue_area = sum(batch_lengths) 
            orange_area = sum(max_length - length for length in batch_lengths)
            ratio = orange_area / (blue_area + orange_area)
            ratios.append(ratio)

            # Highlight the padded parts
            for j in range(start_idx, end_idx): ax.bar(j, max_length - batch_lengths[j - start_idx], bottom=batch_lengths[j - start_idx], color='orange', edgecolor='orange')

            # Add vertical red dashed lines at batch boundaries
            if i > 0: ax.axvline(x=start_idx, color='red', linestyle='--')

        ratio = np.mean(ratios)
        print(f"{descr} = {ratio*100:.0f} %")

        # Set labels and title
        ax.set_xlabel('Sequence Number')
        ax.set_ylabel('Sequence Length (Number of Rows)')
        #ax.set_title('Diagram of Sequence Lengths with Padded Parts Highlighted')
        handles = [plt.Rectangle((0,0),1,1,color=c,ec="k") if c != "red" else plt.Line2D([0], [0], color=c, linestyle='--') for c in ["blue", "orange", "red"]]
        labels = ["sequence data", "padding", "batch boundaries"]
        ax.legend(handles, labels, loc="upper right"); ax.grid(False); plt.show();

        return ratio

# SEQUENCE PADDING: VISUALIZATION -------------------------------------------
def visualize_padding(BATCH_SIZE, trip_lengths, sorted_trip_lengths, train_loader, val_loader, test_loader):
    # compare padding proportions for unsorted and sorted sequences, as well as for the train, val, and test sets
    _ = plot_padded_sequences(BATCH_SIZE, trip_lengths, "padding values (unsorted)")
    _ = plot_padded_sequences(BATCH_SIZE, sorted_trip_lengths, "padding values (sorted)")
    _ = plot_padded_sequences(BATCH_SIZE, get_trip_lengths_from_loader(train_loader), "padding values (Train Set)")
    _ = plot_padded_sequences(BATCH_SIZE, get_trip_lengths_from_loader(val_loader), "padding values (Val Set)")
    _ = plot_padded_sequences(BATCH_SIZE, get_trip_lengths_from_loader(test_loader), "padding values (Test Set)")


# GET SEQUENCE LENGTHS FROM BATCH_DATALOADER OBJECT (for plotting only)  ---------------------------------------------------------------------
def get_trip_lengths_from_loader(data_loader) -> list:
    """
    Extracts and processes trip lengths from a data loader.
    This function iterates through the provided data loader to extract trip lengths,
    moves the shortest batch to the end of the list, and returns a concatenated list
    of all trip lengths.
    Args:
        data_loader (DataLoader): A PyTorch DataLoader object that yields batches of data.
            Each batch is expected to be a tuple where the third element contains the lengths
            of the trips.
    Returns:
        list: A concatenated list of all trip lengths, with the shortest batch moved to the end.
    """
    # obtain the sequences from train_loader for plotting
    trip_lengths = []
    for _, _, lengths in data_loader:
        trip_lengths.append(lengths.tolist())

    # Move shortest batch to the end:
    incomplete_batch = min(trip_lengths, key=len)
    trip_lengths.remove(incomplete_batch)
    trip_lengths.append(incomplete_batch)

    return list(np.concatenate(trip_lengths))

###################################################################################################################################