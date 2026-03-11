import heapq
import random
import threading
import time
from typing import List, Set

import bittensor as bt

from zeus.base.validator import BaseValidatorNeuron
from zeus.utils.uids import get_available_uids, get_random_uids
from zeus.validator.constants import MAINNET_UID, PERMITTED_MINER_STRIKES, FORWARD_RESPONSE_BATCH_K


class UIDTracker:

    def __init__(self, validator: BaseValidatorNeuron):
        self.validator = validator
        self._busy_uids = set()
        self._last_good_uids = set()
        self.lock = threading.Lock()

    
    def init_count_map(self):
        self.count_map = {}

    def get_next_batch(self, k: int, good_miners_uids: Set[int], trial_num: int, ignore_busy_after_step: int, allowed_uids: Set[int] = None) -> List[int]:
        
        all_avail_miner_uids = get_available_uids(
            self.validator.metagraph,
            self.validator.config.neuron.vpermit_tao_limit,
            MAINNET_UID, 
            exclude=good_miners_uids
        )
        if allowed_uids is not None:
            all_avail_miner_uids = [uid for uid in all_avail_miner_uids if uid in allowed_uids]

        #busy_uids = self.get_busy_uids() if trial_num < ignore_busy_after_step else set()
        #bt.logging.warning(f"get_next_batch: busy_uids: {busy_uids}")
        # Create a priority queue: (attempts, uid)
        # Only include UIDs that are technically available and not 'good' or 'busy'
        priority_queue = []
        for uid in all_avail_miner_uids:
            uid_strike = self.count_map.get(uid, 0)
            if uid_strike < PERMITTED_MINER_STRIKES:
                heapq.heappush(priority_queue, (uid_strike, uid))

        selected = []
        while len(selected) < k and priority_queue:
            attempts, uid = heapq.heappop(priority_queue)
            selected.append(uid)
            # Increment the count for next time
            self.count_map[uid] = attempts + 1
        

        return selected



