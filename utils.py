import torch

def get_device():
    """
    Returns the appropriate torch.device:
    - 'cuda' if CUDA is available.
    - 'mps' if on a Mac with Metal Performance Shaders (MPS) support.
    - 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()