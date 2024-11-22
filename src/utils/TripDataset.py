import torch
from torch.utils.data import Dataset
import pandas as pd

# DATASET DEFINITION -----------------------------------------------------------------------
class TripDataset(Dataset):
    def __init__(self, file_list, input_columns, target_column, scaler, target_scaler, fit=False):
        self.file_list = file_list
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.fit = fit
        self.data = []
        self.targets = []

        # fitting scalers over complete training dataset
        if self.fit:
            print(f"fitting Scalers: {scaler.__class__.__name__}, {target_scaler.__class__.__name__}")
            # Initialize and Fit the scalers on the complete training data set
            # Fit the scalers incrementally to avoid memory errors
            for file in self.file_list:
                df = pd.read_parquet(file, columns = input_columns+[target_column], engine='fastparquet')
                X = df[input_columns].values
                y = df[target_column].values.reshape(-1, 1)  # Reshape to match the shape of the input
                self.scaler.partial_fit(X)
                self.target_scaler.partial_fit(y)
            print(f"Done. Create DataSets...")

        # transform with fitted scalers
        for file in self.file_list:
            # DATA PREPROCESSING -----------------------------------------------------------
            # Assigning inputs and targets and reshaping ---------------
            df = pd.read_parquet(file, columns = input_columns+[target_column], engine='fastparquet')
            X = df[input_columns].values
            y = df[target_column].values.reshape(-1, 1)  # Reshape to match the shape of the input
            # use the previously fitted scalers to transform the data
            X = self.scaler.transform(X)  
            y = self.target_scaler.transform(y).squeeze()
            # Append to data
            self.data.append(X)
            self.targets.append(y.squeeze())

    def __len__(self):
        return sum(len(target) for target in self.targets)

    def __getitem__(self, index):
        # Find which file the index belongs to
        # enables indexing over concatenated dataset via one timestep index
        for i, target in enumerate(self.targets):
            if index < len(target):
                return (
                    torch.tensor(self.data[i][index], dtype=torch.float32).unsqueeze(0),  # Add time dimension
                    torch.tensor(target[index], dtype=torch.float32)
                )
            index -= len(target)
        raise IndexError("Index out of range")
