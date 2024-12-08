import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    def __init__(self, file_list, scaler):
        self.file_list = file_list
        self.scaler = scaler
        
        self.data = []
        self.targets = []

        for file in self.file_list:
            df = pd.read_parquet(file, columns=columns)
            df.fillna(df.median(), inplace=True)  # Basic Median Filling 
            
            X = df[input_columns].values
            y = df[target_column].values
            
            # Normalize inputs
            X = self.scaler.fit_transform(X)
            
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

folder_path = "data/processed"

# Only 100 Input-files for testing
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")][:100]


train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_dataset = TripDataset(train_files, scaler)
test_dataset = TripDataset(test_files, scaler)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

model = LSTMModel(input_size=len(input_columns))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 500

# Print model information
print(model)

print(f"\nModel state:")
for param_tensor in model.state_dict():
    print(f"{param_tensor}: \t{model.state_dict()[param_tensor].shape}")
    

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad() 
        outputs = model(inputs) 
        loss = criterion(outputs.squeeze(), targets) 
        loss.backward() 
        optimizer.step()  
        
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

model.eval()
test_loss = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
