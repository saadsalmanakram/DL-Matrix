from torch_geometric.models import GCN

# Load a pretrained GCN model (hypothetical example)
pretrained_model = GCN(in_channels=1, hidden_channels=16, out_channels=2)
pretrained_model.load_state_dict(torch.load('pretrained_gcn.pth'))

# Use the pretrained model
pretrained_model.eval()
with torch.no_grad():
    out = pretrained_model(data)
    print(out)
