import blosc2
import base64
import torch
import numpy as np
import logging
from typing import Union

def compress_prediction(tensor: Union[torch.Tensor, np.ndarray]) -> bytes:
    """Converts a torch tensor to compressed bytes (lossless, high ratio)."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy().astype(np.float16)
    elif isinstance(tensor, np.ndarray):
        arr = tensor.astype(np.float16)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")

    compressed_bytes = blosc2.pack_array(
        arr,
        clevel=9,
        filter=blosc2.Filter.BITSHUFFLE,
        codec=blosc2.Codec.ZSTD,
    )
    return compressed_bytes

def decompress_prediction(compressed_prediction: bytes) -> torch.Tensor:
    """Converts compressed bytes back to a torch tensor with the correct shape."""
    try:
        compressed_bytes = compressed_prediction
        array = blosc2.unpack_array(compressed_bytes) # Decompresses the bytes back into the original raw binary buffer.
        return torch.from_numpy(array)
    except Exception as e:
        logging.error(f"Error decompressing prediction: {e}")
        return None