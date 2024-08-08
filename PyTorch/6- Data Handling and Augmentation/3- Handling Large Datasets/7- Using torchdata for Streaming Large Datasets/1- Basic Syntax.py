import torch
from torchdata.datapipes.iter import FileLister, FileOpener

# Example file listing and opening
file_lister = FileLister(root='large_dataset_directory', masks='*.pt')
file_opener = FileOpener(file_lister, mode='rb')

# Stream processing
for file in file_opener:
    tensor = torch.load(file[1])
    # Process the tensor
    pass
