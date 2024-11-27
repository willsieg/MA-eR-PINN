import torch
import bisect
import random
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# -----------------------------------------------------------------------------------------------------------
# DATASET DEFINITION (SEQUENCE LEVEL) -----------------------------------------------------------------------
class TripDataset(Dataset):
    def __init__(self, file_list: list, input_columns: list, target_column: str, scaler, target_scaler, fit: bool = False):
        self.file_list = file_list
        self.input_columns = input_columns
        self.target_column = target_column
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.fit = fit
        self.data = []
        self.targets = []
        self.cumulative_lengths = []

        # fitting scalers over complete training dataset
        if self.fit:
            print(f"fitting Scalers: {scaler.__class__.__name__}, {target_scaler.__class__.__name__}")
            # Initialize and Fit the scalers on the complete training data set
            # Fit the scalers incrementally to avoid memory errors
            num_files = len(self.file_list)
            for i, file in enumerate(self.file_list):
                df = pd.read_parquet(file, columns = input_columns+target_column, engine='fastparquet')
                X = df[input_columns].values
                y = df[target_column].values.reshape(-1, 1)  # Reshape to match the shape of the input: 2D array with one column
                self.scaler.partial_fit(X)
                self.target_scaler.partial_fit(y)
            
                # Print status info at 50%
                if i == num_files // 2: print(f"\t50% of the fitting done...")
                
            print(f"Done. Create DataSets and DataLoaders...")

        # transform with fitted scalers
        cumulative_length = 0
        for i, file in enumerate(self.file_list):
            # DATA PREPROCESSING -----------------------------------------------------------
            # Assigning inputs and targets and reshaping ---------------
            df = pd.read_parquet(file, columns = input_columns+target_column, engine='fastparquet')
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

# -----------------------------------------------------------------------------------------------------------
# DATASET DEFINITION (BATCH LEVEL) -----------------------------------------------------------------------
class BatchDataset(Dataset):
    def __init__(self, batches: list):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int):
        return self.batches[idx]


def create_batches(dataset: Dataset, batch_size: int, shuffle_batches: bool = False) -> BatchDataset:
    # Create a list to store the batches
    batches = []

    # Iterate through the sorted dataset in chunks of batch_size
    for i in range(0, len(dataset.targets), batch_size):
        batch = [(dataset.data[j], dataset.targets[j]) for j in range(i, min(i + batch_size, len(dataset.targets)))]
        batches.append(batch)

    # Shuffle the order of batches
    if shuffle_batches: random.shuffle(batches)

    # Print the number of batches created
    print(f"Number of batches created: {len(batches)}")
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


# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
