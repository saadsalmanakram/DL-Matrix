from torch.utils.data import SequentialSampler, DataLoader

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

for batch in dataloader:
    data, targets = batch
    print(data, targets)
