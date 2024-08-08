# Apply gradient clipping during training loop
for p in model.parameters():
    p.grad.data.clamp_(-1, 1)
