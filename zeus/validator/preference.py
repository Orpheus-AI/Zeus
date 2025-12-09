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

from typing import List, Optional, Tuple
import numpy as np
import bittensor as bt

from zeus.utils.misc import copy_fitting
from zeus.utils.uids import is_miner_uid
from zeus.base.dendrite import ZeusDendrite
from zeus.protocol import PreferenceSynapse
from zeus.validator.constants import (
    MechanismType, 
    PREFERENCE_UPDATE_FREQUENCY,
)

class PreferenceManager:

    # default to ERA5 if unspecified
    UNSPECIFIED = MechanismType.ERA5.value

    def __init__(
        self,
        num_neurons: int,
        frequency: int = PREFERENCE_UPDATE_FREQUENCY,
    ):
        self.last_queried = 0 # always query on first forward
        self.frequency = frequency
        # default to -1 in case never set
        self.preferences = np.full(num_neurons, fill_value=self.UNSPECIFIED)
        self.to_query = set()

    def should_query(self, block: int) -> bool:
        """
        Check if we should query the preferences.
        """	
        return (
            (block - self.last_queried) >= self.frequency
            or len(self.to_query) > 0
            )
    
    def mark_for_query(self, uids: List[int]):
        """
        Mark the uids for query. Note that we can call this multiple times
        """
        self.to_query.update(uids)
    
    def load_preferences(self, preferences: np.ndarray):
        self.preferences = copy_fitting(preferences, self.preferences)

    def get_preferences(self) -> np.ndarray:
        return self.preferences

    def get_preference(self, uid: int) -> Optional[MechanismType]:
        """
        Get the preference of a uid.
        Returns:
            Optional[MechanismType]: The preference.
        """
        try:
            return MechanismType(self.preferences[uid])
        except:
            return None

    def reshape_preferences(self, new_size: int):
        """
        Reshape the preferences if the metagraph size changes.
        """
        old_size = len(self.preferences)
        self.preferences = copy_fitting(self.preferences, np.full(new_size, fill_value=self.UNSPECIFIED))

        new_uids = np.arange(new_size)[old_size:]
        self.to_query.update(new_uids)

    
    async def query_preferences(self, dendrite: ZeusDendrite, metagraph: bt.Metagraph) -> Tuple[List[int], List[int]]:
        """
        Query the preferences from the dendrite.
        Note that we only query uids that are not validators. 
        If to_query is not empty, we query those uids, otherwise we query all uids that are not validators.
        Only sets preferences for uids that respond.

        Returns:
            List[int]: The uids that responded and have a preference set.
            List[int]: The uids that did not respond or responded with an invalid preference.
        """
        self.last_queried = metagraph.block
        uids = self.to_query or [n.uid for n in metagraph.neurons if is_miner_uid(metagraph, n.uid)]
        self.to_query = set()

        axons = [metagraph.axons[uid] for uid in uids]
        responses = await dendrite(
            axons=axons,
            synapse=PreferenceSynapse(),
            deserialize=True,
            timeout=10,
            semaphore=150,
        )

        specified_uids, invalid_uids = [], []
        for uid, response in zip(uids, responses):
            try:
                self.preferences[uid] = MechanismType(response)
                specified_uids.append(uid)
            except:
                invalid_uids.append(uid)

        return specified_uids, invalid_uids

