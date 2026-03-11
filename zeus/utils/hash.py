# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Ørpheus A.I.

import hashlib
import base64
import time
import bittensor as bt
from zeus.utils.compression import compress_prediction


def prediction_hash(compressed_bytes: bytes, hotkey: str) -> str:
    """
    Canonical hash for validator-miner commitment: sha256(compressed_predictions + hotkey).
    Used for HashedTimePredictionSynapse and verification.
    """
    if compressed_bytes is None:
        bt.logging.warning("Compressed bytes are None, returning None")
        return None
    hotkey_bytes = hotkey.encode("utf-8")
    # note that sha256 takes bytes, so we need to encode the hotkey to bytes
    return hashlib.sha256(compressed_bytes + hotkey_bytes).hexdigest()

