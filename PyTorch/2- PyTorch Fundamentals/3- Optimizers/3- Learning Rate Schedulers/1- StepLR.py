# Decays the learning rate by a factor every few epochs.

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
