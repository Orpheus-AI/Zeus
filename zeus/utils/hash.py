# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Ørpheus A.I.

import hashlib
import bittensor as bt


def prediction_hash(compressed_bytes: bytes, hotkey: str) -> str:
    """
    Canonical hash for validator-miner commitment: sha256(compressed_predictions + hotkey).
    Used for HashedTimePredictionSynapse and verification.
    """
    if compressed_bytes is None:
        bt.logging.warning("Compressed bytes are None, returning None")
        return None
    # note that sha256 takes bytes, so we need to encode the hotkey to bytes
    hotkey_bytes = hotkey.encode("utf-8")
    return hashlib.sha256(compressed_bytes + hotkey_bytes).hexdigest()

