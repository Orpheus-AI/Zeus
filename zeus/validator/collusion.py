from typing import List, Dict
from zeus.validator.miner_data import MinerData
import bittensor as bt
from itertools import combinations
from zeus.validator.constants import COLLUSION_PENALTY_THRESHOLD, SHORT_CHALLENGE, LONG_CHALLENGE

def metric_similarity(val1: float, val2: float) -> float:
    """Calculates similarity dynamically based on the passed metric name."""
    return abs(val1 - val2) / (max(val1, val2) + 1e-9)

def are_colluding(val1: float, val2: float, threshold: float) -> bool:
    
    return metric_similarity(val1, val2) < threshold

def order_pair_by_registration(m1: MinerData, m2: MinerData, hotkey2registration_block: Dict[str, int]):
    """Returns (newest, oldest) based on registration block."""
    # Fixed the return syntax here to ensure it correctly returns a 2-item tuple
    if hotkey2registration_block[m1.hotkey] > hotkey2registration_block[m2.hotkey]:
        return m1, m2
    return m2, m1

def pick_threshold(predict_hours: int) -> float:
    if predict_hours <= 49:
        return COLLUSION_PENALTY_THRESHOLD[SHORT_CHALLENGE]
    else:
        return COLLUSION_PENALTY_THRESHOLD[LONG_CHALLENGE]

def apply_collusion_penalty(miners_data: List[MinerData], hotkey2registration_block: Dict[str, int], threshold: float) -> List[MinerData]:
    """
    Identifies collusion. If two miners are too similar, the NEWER registration 
    (higher UID) is penalized, keeping the OLDEST (lower UID).
    """
    # 1. Quick filter for valid miners using list comprehension
    active = sorted([m for m in miners_data if m.rmse is not None and m.rmse < float('inf')], key=lambda x: x.uid)
    if not active: return miners_data

    penalized_uids = set()

    rmse_metric_name = "rmse"
    for m_new, m_old in combinations(active, 2):
        if m_new.uid in penalized_uids or m_old.uid in penalized_uids:
            continue

        val1 = getattr(m_new, rmse_metric_name)
        val2 = getattr(m_old, rmse_metric_name)
        rmse_colluding =  are_colluding(val1, val2, threshold)
        if rmse_colluding:
            newest, oldest = order_pair_by_registration(m_new, m_old, hotkey2registration_block)
            bt.logging.warning(f"Collusion detected {val1} and {val2}! Keeping {oldest.uid} and penalizing {newest.uid}")
            # Bulk update attributes
            newest.rmse = newest.mae = newest.score = None
            newest.shape_penalty = True
            penalized_uids.add(newest.uid)

    bt.logging.warning(f'{miners_data}') # TODO maybe comment this out

    return miners_data

