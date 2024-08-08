# Decays the learning rate by a factor of gamma every epoch.

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
