from torch.utils.data import IterableDataset

class MyIterableDataset(IterableDataset):
    def __iter__(self):
        # Define how to iterate over the dataset
        for i in range(10):
            yield i

dataset = MyIterableDataset()
dataloader = DataLoader(dataset, batch_size=2)

for batch in dataloader:
    print(batch)
