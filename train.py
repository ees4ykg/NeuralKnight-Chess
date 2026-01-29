import os
import numpy as np 
import time
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import h5py

# Load the move_to_idx mapping
with open("models/model3_move_to_idx", "rb") as file:
    move_to_idx = pickle.load(file)

# Load data into RAM
print("Loading data into RAM...")
with h5py.File("synthetic_data/stockfish_generated_data.h5", 'r') as f:
    X_tensor = torch.from_numpy(f['positions'][:]).float()
    moves = [m.decode('utf-8') for m in f['moves'][:]]
    y_tensor = torch.tensor([move_to_idx[m] for m in moves], dtype=torch.long)

print(f"Dataset size: {len(X_tensor)} positions")
print(f"X shape: {X_tensor.shape}, memory: {X_tensor.element_size() * X_tensor.nelement() / 1e9:.2f} GB")

# Create Dataset and DataLoader
from torch_model import NeuralKnightChessModel2

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Initialization
model = NeuralKnightChessModel2(num_classes=len(move_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    end_time = time.time()
    epoch_time = end_time - start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time) - minutes * 60
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"models/model3_checkpoint_epoch{epoch+1}.pth")

# Save the final model
torch.save(model.state_dict(), "models/100_epochs_model3.pth")