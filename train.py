import os
import numpy as np 
import time
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from chess import pgn 
from tqdm import tqdm


# Load the data
x_data = np.load("cleaned_data/x_data.npy")  # Shape: (num_samples, 14, 8, 8)
y_data = np.load("cleaned_data/y_data.npy")  # Shape: (num_samples,)



# Convert directly to tensors
X_tensor = torch.from_numpy(x_data).float()
y_tensor = torch.from_numpy(y_data).long()

print(X_tensor.shape)  # torch.Size([num_samples, 14, 8, 8])
print(y_tensor.shape)  # torch.Size([num_samples])


# Create Dataset and DataLoader
from torch_dataset import NeuralKnightChessDataset
from torch_model import NeuralKnightChessModel

dataset = NeuralKnightChessDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Initialization
model = NeuralKnightChessModel(num_classes=len(y_tensor.unique())).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)  # Raw logits

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    end_time = time.time()
    epoch_time = end_time - start_time
    minutes: int = int(epoch_time // 60)
    seconds: int = int(epoch_time) - minutes * 60
    print(f'Epoch {epoch + 1 + 50}/{num_epochs + 1 + 50}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')

# Save the model
torch.save(model.state_dict(), "models/100_epochs_model2.pth")

