# Adjusts the learning rate using a cosine annealing schedule.

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
