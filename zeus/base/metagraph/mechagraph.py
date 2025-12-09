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

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import bittensor as bt
import numpy as np
from zeus.validator.constants import MechanismType, VPERMIT_TAO_LIMIT
from zeus.utils.uids import get_uids
if TYPE_CHECKING:
    from zeus.base.metagraph.delegate import MetagraphDelegate


class Mechagraph(bt.Metagraph):

    def __init__(
        self,
        size: int,
        parent: MetagraphDelegate,
        mechanism: MechanismType,
    ):
        # Disable sync here since parent will sync it already
        super().__init__(parent.netuid, parent.network, parent.lite, False, parent.subtensor, mechanism.value)
        self.mechanism = mechanism
        self.parent = parent
        self.size = size

    def sync(*_):
        """
        Prevent accidental use of this method without its parent.
        """
        raise NotImplementedError("Use parent MetagraphDelegate to sync mechagraphs!.")

    def _sync(
        self,
        block: int,
        lite: Optional[bool] = None,
    ):
        """
        Syncs based on parent metagraph. 
        Different name to prevent it being accidentally called without its parent
        """
        self._assign_neurons()
        # Set attributes for metagraph
        self._set_metagraph_attributes(block)

        # Doesn't actually require a subtensor or block since we aren't root network
        if not lite:
            self._set_weights_and_bonds(block, None)

        self._get_all_stakes_from_chain()
        # block not required anymore
        self._apply_extra_info(None)

    def _assign_neurons(self, *_):
        """
        Assigns neurons to the metagraph based on the provided parent neurons
        Filter only those with highest incentive for this mechanism.
        NOTE: BT's default version does not work, so fetch incentives separately.
        Allow varargs inputs to match signature, but not used.
        """
        self.mechagraph_info = self.parent.subtensor.get_metagraph_info(netuid=self.netuid, mechid=self.mechid)
        # set for use in _get_all_stakes_from_chain and _apply_extra_info
        incentives = np.asarray(self.mechagraph_info.incentives)
        vali_or_nonserve_uids = get_uids(
            metagraph=self.parent,
            vpermit_tao_limit=VPERMIT_TAO_LIMIT,
            miners=False,
        )
        try:
            incentives[vali_or_nonserve_uids] = 0.
        except IndexError:
            bt.logging.error(f"Failed to filter validator or non-serving neurons for mechagraph {self.mechid}")
            
        parent_idxs = np.argpartition(incentives, -self.size)[-self.size:]
        self.parent_idxs = np.sort(parent_idxs)

        mech_neurons = []
        for idx in self.parent_idxs:
            neuron = self.parent.neurons[idx]
            neuron.incentive = self.mechagraph_info.incentives[idx]
            mech_neurons.append(neuron)

        self.neurons = mech_neurons


    def _get_all_stakes_from_chain(self, *_):
        """
        Fills in the stake associated attributes of a class instance from parent.
        Allow varargs inputs to match signature, but not used."""
        self.alpha_stake = self.parent.alpha_stake[self.parent_idxs]
        self.tao_stake = self.parent.tao_stake[self.parent_idxs]
        self.total_stake = self.parent.total_stake[self.parent_idxs]
        

    def _apply_extra_info(self, *_):
        """Applies mechagraph info fetched in _assign_neurons."""
        self._apply_metagraph_info_mixin(metagraph_info=self.mechagraph_info)
        # correct these manually
        self.num_uids = len(self.neurons)
        self.max_uids = self.size
        
        # these are otherwise set for whole graph again so filter out
        self.identities = np.array(self.identities)[self.parent_idxs].tolist()
        self.pruning_score = np.array(self.pruning_score)[self.parent_idxs].tolist()
        self.block_at_registration = np.array(self.block_at_registration)[self.parent_idxs].tolist()
        self.tao_dividends_per_hotkey = np.array(self.tao_dividends_per_hotkey)[self.parent_idxs].tolist()
        self.alpha_dividends_per_hotkey = np.array(self.alpha_dividends_per_hotkey)[self.parent_idxs].tolist()