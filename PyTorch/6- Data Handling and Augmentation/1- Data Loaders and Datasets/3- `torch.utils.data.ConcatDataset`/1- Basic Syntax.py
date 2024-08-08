from torch.utils.data import ConcatDataset

# Assume `dataset1` and `dataset2` are instances of `torch.utils.data.Dataset`
combined_dataset = ConcatDataset([dataset1, dataset2])

dataloader = DataLoader(combined_dataset, batch_size=2)
