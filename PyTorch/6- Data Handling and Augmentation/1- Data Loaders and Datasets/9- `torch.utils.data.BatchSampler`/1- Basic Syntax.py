from torch.utils.data import BatchSampler, RandomSampler, DataLoader

batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=2, drop_last=False)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

for batch in dataloader:
    data, targets = batch
    print(data, targets)
