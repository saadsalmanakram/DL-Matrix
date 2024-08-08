import torch.multiprocessing as mp

def train(rank, world_size):
    # Training logic for each process
    pass

if __name__ == "__main__":
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
