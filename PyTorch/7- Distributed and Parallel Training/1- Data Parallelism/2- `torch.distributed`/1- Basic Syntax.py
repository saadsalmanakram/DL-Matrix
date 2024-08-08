import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Initialize the process group for distributed training
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Define your model and use DistributedDataParallel
def main(rank, world_size):
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss().to(rank)

    # Distributed data parallelism setup
    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)

    # Training loop
    for inputs, labels in dataloader:
        inputs = inputs.to(rank)
        labels = labels.to(rank)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    world_size = 2  # Number of GPUs/nodes
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
