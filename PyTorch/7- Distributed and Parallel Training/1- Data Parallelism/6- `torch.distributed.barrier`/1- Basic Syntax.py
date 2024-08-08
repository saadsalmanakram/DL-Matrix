import torch.distributed as dist

# Initialize the process group
dist.init_process_group(backend='nccl', init_method='env://', world_size=4, rank=0)

# Synchronize processes
dist.barrier()

# Continue with further computation
