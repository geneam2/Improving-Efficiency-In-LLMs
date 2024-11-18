from utils import device
import torch

# Example usage
print(f"Using device: {device}")

# Creating tensors on the appropriate device
tensor = torch.tensor([1, 2, 3]).to(device)
print(tensor)