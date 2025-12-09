# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Set
import time
from collections import defaultdict
import threading
import random

import numpy as np
import bittensor as bt
from zeus.base.validator import BaseValidatorNeuron
from zeus.validator.preference import PreferenceManager
from zeus.validator.constants import MechanismType
import zeus.utils.uids as uid_utils

class UIDTracker:

    def __init__(self, validator: BaseValidatorNeuron, preference_manager: PreferenceManager):
        self.validator = validator
        self.preference_manager = preference_manager
        self._busy_uids = defaultdict(set)
        self._last_good_uids = defaultdict(set)
        self.lock = threading.Lock()    

    def get_uids(self, mechanism: MechanismType, exclude: Set[int]) -> List[int]:
        """
        Get all uids that have set a preference for the given mechanism.
        Args:
            mechanism: The mechanism to get the uids for.
            exclude: The uids to exclude from the selection.

        Returns:
            List[int]: The uids that are available for the given mechanism.
        """
        avail_uids = []
        for uid in uid_utils.get_uids(
            metagraph=self.validator.metagraph,
            vpermit_tao_limit=self.validator.config.neuron.vpermit_tao_limit,
            miners=True,
        ):
            if self.preference_manager.get_preference(uid) != mechanism:
                continue

            if uid not in exclude:
                avail_uids.append(uid)
        return avail_uids

    def get_random_uids(self, k: int, mechanism: MechanismType, tries: int = 3, sleep: int = 1) -> List[int]:
        """
        Get k random uids that have set a preference for the given mechanism.
        Note that we only sample uids that are not busy, except for the last attempt.
        If we fail to sample enough uids, we retry up to tries times.

        Args:
            k: The number of uids to get.
            mechanism: The mechanism to get the uids for.
            tries: The number of tries to get the uids.
            sleep: The sleep time between tries.

        Returns:
            List[int]: The uids that were sampled.
        """

        attempt = 1
        while True:
            if attempt > 1:
                 # sleep here so no delay once we sample miners to add them to our busy-list
                 bt.logging.warning(f"Failed to sample enough non-busy miner uids, retrying in {sleep} second(s). ATTEMPT {attempt}/{tries}")
                 time.sleep(sleep)

            avail_uids = self.get_uids(
                mechanism, 
                exclude=self.get_busy_uids(mechanism) if attempt < tries else set()
            )

            sample_size = min(k, len(avail_uids))
            if sample_size == 0:
                miner_uids = np.array([], dtype=int)
            miner_uids = np.array(random.sample(avail_uids, sample_size))

            if len(miner_uids) == k or attempt == tries:
                self.add_busy_uids(miner_uids, mechanism)
                return miner_uids

            attempt += 1
    
    # NOTE: since this is access by proxy and forward, need a thread lock!
    def get_busy_uids(self, mechanism: MechanismType) -> Set[int]:
        """
        Get the uids that are busy for the given mechanism.
        Args:
            mechanism: The mechanism to get the busy uids for.

        Returns:
            Set[int]: The uids that are busy for the given mechanism.
        """

        with self.lock:
            return self._busy_uids[mechanism]
    
    def add_busy_uids(self, uids: List[int], mechanism: MechanismType):
        """
        Add the uids to the busy list for the given mechanism.
        Args:
            uids: The uids to add to the busy list.
            mechanism: The mechanism to add the uids to.
        """

        with self.lock:
            self._busy_uids[mechanism].update(set(uids))

    def mark_finished(self, uids: List[int], mechanism: MechanismType, good: bool = False):
        """
        Mark the uids as finished for the given mechanism.
        Args:
            uids: The uids to mark as finished.
            mechanism: The mechanism to mark the uids as finished for.
            good: If the uids are 'good', they will be added to the last good uids list for the API Proxy.
        """
        with self.lock:
            self._busy_uids[mechanism] = self._busy_uids[mechanism] - set(uids)
            if good:
                self._last_good_uids[mechanism] = set(uids)

    def get_responding_uids(self, mechanism: MechanismType) -> Set[int]:
        """
        Get the uids that are responding for the given mechanism for proxy and not busy.
        Args:
            k: The number of uids to get.
            mechanism: The mechanism to get the responding uids for.

        Returns:
            List[int]: The uids that are responding and not busy for the given mechanism.
        """
        with self.lock:
            self._last_good_uids[mechanism] = self._last_good_uids[mechanism] - self._busy_uids[mechanism]
        return self._last_good_uids[mechanism]
