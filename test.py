import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())

