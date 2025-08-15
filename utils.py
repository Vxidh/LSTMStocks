import gc
import torch

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
