import heapq
import threading
from typing import List, Set

import bittensor as bt

from zeus.base.validator import BaseValidatorNeuron
from zeus.utils.uids import get_available_uids
from zeus.validator.constants import MAINNET_UID

def shift_uids_past_validator(sorted_uids: List[int], validator_uid: int) -> List[int]:
    """
    Reorders UIDs so the sequence starts immediately after the validator's own ID.
    This 'rotates' the UID circle so different validators start their search 
    at different points, preventing all validators from hitting UID 0 first.
    """
    cut_index = next((idx for idx, u in enumerate(sorted_uids) if u > validator_uid), 0)
    return sorted_uids[cut_index:] + sorted_uids[:cut_index]


class UIDTracker:
    def __init__(self, validator: BaseValidatorNeuron):
        self.validator = validator
        self._busy_uids = set()
        self._last_good_uids = set()
        self.lock = threading.Lock()

    def init_count_map(self):
        # Stores how many times each UID has been attempted (the 'strike' count)
        self.count_map = {}

    def get_next_batch(self, k: int, good_miners_uids: Set[int], allowed_uids: Set[int] = None, allowed_attempts = 2) -> List[int]:
        """Main entry point: selects the next 'k' miners to query."""
        # Filter out UIDs that are already known to be 'good' or unavailable
        all_avail_miner_uids = self._get_eligible_uids(allowed_uids, excluded_uids=good_miners_uids)
        
        # Sort and rotate the list based on this validator's UID position
        sorted_eligible = self._order_uids(all_avail_miner_uids)
        
        # Build a heap where miners with FEWER attempts/strikes are prioritized
        priority_queue = self._build_priority_queue(sorted_eligible, allowed_attempts)
        
        # Pull the top k miners and increment their strike counts
        selected = self._select_and_update_miners(priority_queue, k)

        return selected
    
    def _get_eligible_uids(self, allowed_uids: list, excluded_uids: list) -> list:
        """Filters the global metagraph for miners meeting specific Tao/Network requirements."""
        all_avail_miner_uids = get_available_uids(
            self.validator.metagraph,
            self.validator.config.neuron.vpermit_tao_limit,
            MAINNET_UID, 
            exclude=excluded_uids
        )
        if allowed_uids is not None:
            all_avail_miner_uids = [uid for uid in all_avail_miner_uids if uid in allowed_uids]
            
        return all_avail_miner_uids
    
    def _order_uids(self, uids: list) -> list:
        """Applies the circle-rotation logic to the list of available UIDs."""
        sorted_uids = sorted(uids)
        return shift_uids_past_validator(sorted_uids, self.validator.uid)

    def _build_priority_queue(self, sorted_eligible: list, allowed_attempts: int) -> list:
        """
        Creates a min-heap. 
        Tuple priority: (strike_count, rotation_index, uid).
        This ensures we try miners with 0 strikes first, then 1, etc.
        """
        priority_queue = []
        for order_index, uid in enumerate(sorted_eligible):
            uid_strike = self.count_map.get(uid, 0)
            if uid_strike < allowed_attempts:
                heapq.heappush(
                    priority_queue,
                    (uid_strike, order_index, uid),
                )
        return priority_queue
    
    def _select_and_update_miners(self, priority_queue: list, k: int) -> list:
        """Extracts miners from heap and updates the count_map to track usage."""
        selected = []
        while len(selected) < k and priority_queue:
            attempts, _, uid = heapq.heappop(priority_queue)
            selected.append(uid)
            # Increment the strike count so they move to the back of the priority line
            self.count_map[uid] = attempts + 1
        return selected
    
    def get_current_strike(self, uids: List[int]):
        """Helper to check how many strikes specific UIDs currently have."""
        if self.count_map:
            return [self.count_map.get(uid, 0) for uid in uids]
        return None