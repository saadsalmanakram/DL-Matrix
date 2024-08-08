from torch.utils.data import RandomSampler, DataLoader

sampler = RandomSampler(dataset)
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

for batch in dataloader:
    data, targets = batch
    print(data, targets)
