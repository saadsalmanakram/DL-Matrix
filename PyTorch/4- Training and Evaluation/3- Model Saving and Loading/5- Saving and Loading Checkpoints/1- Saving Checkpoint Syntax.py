import torch

# Assume 'model' is your PyTorch model and 'optimizer' is your optimizer
checkpoint = {
    'epoch': epoch,  # Replace with the current epoch
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,  # Replace with the current loss value
}
torch.save(checkpoint, 'checkpoint.pth')
