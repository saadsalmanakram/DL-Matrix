import torch.distributed as dist

# Initialize the process group
dist.init_process_group(backend='nccl', init_method='env://', world_size=4, rank=0)

# Create a tensor
tensor = torch.ones(10).cuda()

# Perform all-reduce operation (summing across all processes)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# tensor now contains the sum of the tensors across all processes
