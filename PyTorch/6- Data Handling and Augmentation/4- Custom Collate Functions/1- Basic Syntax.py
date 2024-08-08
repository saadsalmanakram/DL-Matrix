from torch.utils.data import DataLoader

# Custom collate function
def custom_collate_fn(batch):
    data, labels = zip(*batch)
    # Process and stack data and labels here
    return torch.stack(data), torch.tensor(labels)

# Create DataLoader with custom collate function
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_fn)
