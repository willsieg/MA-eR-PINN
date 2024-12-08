import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

columns = ["signal_time", "hirestotalvehdist_cval_icuc", "vehspd_cval_cpc", 
           "altitude_cval_ippc", "bs_roadincln_cval", "ambtemp_cval_pt", 
           "hv_batpwr_cval_bms1", "hv_batmomavldischrgen_cval_1"]

# Time and Target column taken out
input_columns = columns[1:-1]
target_column = columns[-1]


class TripDataset(Dataset):
    def __init__(self, file_list, scaler, target_scaler):
        self.file_list = file_list
        self.scaler = scaler
        self.target_scaler = target_scaler
        
        self.data = []
        self.targets = []

        for file in self.file_list:
            df = pd.read_parquet(file, columns=columns)
            df.fillna(df.median(), inplace=True)  # Basic Median Filling 
            
            X = df[input_columns].values
            y = df[target_column].values.reshape(-1, 1)
            
            # Normalize inputs
            X = self.scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y).squeeze()
            
            # Append to data
            self.data.append(X)
            self.targets.append(y)

    def __len__(self):
        return sum(len(target) for target in self.targets)

    def __getitem__(self, index):
        # Find which file the index belongs to
        for i, target in enumerate(self.targets):
            if index < len(target):
                return (
                    torch.tensor(self.data[i][index], dtype=torch.float32).unsqueeze(0),  # Add time dimension
                    torch.tensor(target[index], dtype=torch.float32)
                )
            index -= len(target)
        raise IndexError("Index out of range")

folder_path = "datasets/processed"

# Only 100 Input-files for testing
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")][:100]


train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

scaler = StandardScaler()
target_scaler = StandardScaler()
train_dataset = TripDataset(train_files, scaler, target_scaler)
test_dataset = TripDataset(test_files, scaler, target_scaler)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

model = LSTMModel(input_size=len(input_columns))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
# 5. Train the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute the loss directly on the normalized values (without inverse transform)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 6. After training, evaluate the model on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        
        # Compute loss directly on normalized values
        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item()

        # Optional: Inverse-transform outputs and targets for evaluation
        scaled_outputs = target_scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
        scaled_targets = target_scaler.inverse_transform(targets.detach().cpu().numpy().reshape(-1, 1))

        # You can use `scaled_outputs` and `scaled_targets` for error metrics in the original scale if needed

    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
