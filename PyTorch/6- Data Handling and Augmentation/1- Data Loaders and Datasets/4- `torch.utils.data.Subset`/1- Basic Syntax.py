from torch.utils.data import Subset

indices = list(range(0, len(dataset), 2))  # Select every other sample
subset = Subset(dataset, indices)

dataloader = DataLoader(subset, batch_size=2)
