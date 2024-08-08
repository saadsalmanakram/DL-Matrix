import h5py
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.data = self.file['data']
        self.labels = self.file['labels']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Use DataLoader to handle large dataset
dataset = HDF5Dataset('large_dataset.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
