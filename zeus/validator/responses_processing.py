from typing import List

import bittensor as bt

from zeus.base.validator import BaseValidatorNeuron
from zeus.protocol import TimePredictionSynapse
from zeus.utils.compression import decode_base64_to_compressed
from zeus.utils.hash import prediction_hash
from zeus.validator.miner_data import MinerData


def _verify_hashes(axons, compressed_predictions: List[bytes], hashes_list: List[str]) -> List[bool]:
    """Verify that compressed predictions match their committed hashes.
    
    Args:
        axons: List of axon objects corresponding to miners
        compressed_predictions: List of compressed prediction bytes
        hashes_list: List of expected hashes for each prediction
        
    Returns:
        List of verified predictions (bytes) or None for failed verifications
    """
    verified_predictions = []
    bt.logging.debug(f"[_verify_hashes]: commitment_store: {hashes_list}")
    count_success = 0
    for axon, prediction, hash in zip(axons, compressed_predictions, hashes_list):
        hotkey = axon.hotkey

        if not isinstance(hotkey, str): hotkey = str(hotkey)
        computed_hash = prediction_hash(prediction, hotkey) if prediction else None
        
        if prediction and computed_hash == hash:
            bt.logging.debug(f'[_verify_hashes] Successfull verification for hotkey {hotkey}')
            verified_predictions.append(prediction)
            count_success += 1
        else:
            if prediction: 
                bt.logging.warning(f"[_verify_hashes] Hash mismatch for UID {hotkey}. Potential cheating!!!!")
            else:
                bt.logging.info(f"[_verify_hashes] No prediction for hotkey {hotkey}")
            verified_predictions.append(None)

    bt.logging.info(f"[_verify_hashes]: count of successful verifications: {count_success} / length of : {len(verified_predictions)}")
    return verified_predictions

def create_compressed_predictions(responses: List[TimePredictionSynapse]) -> List[bytes]:
    """Extract and decode to bytes to create compressed predictions from miner responses.
    
    Args:
        responses: List of TimePredictionSynapse responses from miners
        
    Returns:
        List of compressed prediction bytes (or None for missing predictions)
    """
    compressed_predictions = []
    count_normal = 0
    for r in responses:
        if getattr(r, "predictions", None) is not None:
            # Decode to bytes
            compressed_predictions.append(decode_base64_to_compressed(r.predictions))
            # FREE THE MEMORY: Delete the base64 string from the synapse
            r.predictions = None 
            count_normal += 1
        else:
            compressed_predictions.append(None)

    bt.logging.debug(f"[create_compressed_predictions]: count of normal responses: {count_normal}")
    return compressed_predictions

def _build_bad_miners_data(
    self: BaseValidatorNeuron,
    uids: set[int],
) -> List[MinerData]:
    """Build MinerData for miners that did not respond (no prediction)."""
    miners_data = [
        MinerData(uid=uid, hotkey=self.metagraph.axons[uid].hotkey, prediction=None, shape_penalty=True)
        for uid in uids
    ]
    return miners_data

