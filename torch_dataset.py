from torch.utils.data import Dataset
import h5py
import torch

class NeuralKnightChessDataset(Dataset):
    def __init__(self, hdf5_path, move_to_idx):
        self.hdf5_path = hdf5_path
        self.move_to_idx = move_to_idx
        
        # Open once to get length, then close
        with h5py.File(hdf5_path, 'r') as f:
            self.length = f['positions'].shape[0]
        
        # Will be opened per-worker in __getitem__
        self.file = None

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Lazy open (needed for multiprocessing with DataLoader)
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
        
        x = self.file['positions'][idx]
        move_uci = self.file['moves'][idx].decode('utf-8')
        y = self.move_to_idx[move_uci]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)