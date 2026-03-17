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

    data = arr.tobytes()
    return blosc2.compress(
        data,
        typesize=2,
        clevel=9,
        filter=blosc2.Filter.BITSHUFFLE,
        codec=blosc2.Codec.ZSTD,
    )

def decompress_prediction(compressed_bytes: bytes, shape: torch.Size) -> torch.Tensor:
    """Converts compressed bytes back to a torch tensor with the correct shape."""
    try:
        raw_buffer = blosc2.decompress(compressed_bytes) # Decompresses the bytes back into the original raw binary buffer.
        return torch.from_numpy(
            np.frombuffer(raw_buffer, dtype=np.float16).copy()
        ).reshape(shape)
    except Exception as e:
        logging.error(f"Error decompressing prediction: {e}")
        return None

def decode_base64_to_compressed(b64_str: str) -> bytes:
    """Decodes the synapse string (b64) into compressed bytes."""
    try:
        return base64.b64decode(b64_str)
    except Exception as e:
        logging.error(f"Error decoding base64 to compressed bytes: {e}")
        return None