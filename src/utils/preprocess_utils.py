import os
import random
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq


def prepare_data(input_folder, pth_folder, max_files, min_seq_length, root):
    
    # PREPARE TRAIN & TEST SET ---------------------------------------------------
    all_files = [Path(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".parquet")]
    if max_files is not None: 
        files = random.sample(all_files, max_files)
    else: 
        files = all_files
    print(f"{'-'*60}\nTotal Files:\t{len(files)}")
    # ---------------------------------------------------
    df = pd.read_parquet(Path(input_folder, random.choice(files)), engine='pyarrow')
    all_signals = df.columns
    assert len(all_signals) == 44

    # FILTER INPUT FILES --------------------------------------------------------
    # generate lengths of all files by reading metadata or using presaved lengths
    try:
        presaved_lengths = pd.read_pickle(Path(root, 'data', 'df_files_lengths.pickle'))
        presaved_lengths = presaved_lengths.set_index('FileName').to_dict()['Length']
        trip_lengths = [presaved_lengths[file.name] for file in files]
    except:
        print(f"{'-'*60}\nObtaining sequence lengths... (may take up to 5 minutes)")
        trip_lengths = [pq.read_metadata(file).num_rows for file in files]

    # discard all items shorter than min_seq_length
    filtered_files = []
    filtered_lengths = []
    for file, length in zip(files, trip_lengths):
        if length > min_seq_length: 
            filtered_files.append(file)
            filtered_lengths.append(length)

    # replace lists with only filtered items
    files = filtered_files
    trip_lengths = filtered_lengths
    print(f"Filtered Files:\t{len(files)}\n{'-'*60}")

    # SORT INPUT FILES BY SEQUENCE LENGTH --------------------------------------
    # this is needed in order to later sort the sequence by their length
    file_length_mapping = sorted([(file.name, length, idx) for idx, (file, length) in enumerate(zip(files, trip_lengths))], \
        key=lambda x: x[1], reverse=True)

    file_length_df = pd.DataFrame(file_length_mapping, columns=['FileName', 'Length', 'Index'])
    indices_by_length = file_length_df['Index'].to_list()
    sorted_trip_lengths = file_length_df['Length'].to_list()
    print(file_length_df)
    
    return files, trip_lengths, indices_by_length, sorted_trip_lengths, all_signals


def print_dataset_sizes(train_dataset, val_dataset, test_dataset, train_subset, val_subset, test_subset, files):
    print(f"{'-'*60}\nTrain size:  {len(train_dataset)}\t\t(Files: {len(train_subset)})")
    print(f'Val. size:   {len(val_dataset)}\t\t(Files: {len(val_subset)})')
    print(f'Test size:   {len(test_dataset)}\t\t(Files: {len(test_subset)}) \n {"-"*60}')

    if train_dataset.__len__() != sum(len(data) for data in train_dataset.data): 
        print("Warning: Train Dataset Length Mismatch")
    if len(train_subset) + len(val_subset) + len(test_subset) != len(files): 
        print(f"\tRemoved {len(files) - (len(train_subset) + len(val_subset) + len(test_subset))} file from the dataset\n{'-'*60}")
    
    subset_files = {
        "train_files": list(train_dataset.file_list), 
        "val_files": list(val_dataset.file_list), 
        "test_files": list(test_dataset.file_list)
    }
    
    print(f"first 3 train files: {[os.path.basename(_) for _ in subset_files['train_files'][:3]]}")
    return subset_files