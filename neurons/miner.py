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

import base64
import time
import typing
from datetime import datetime

import bittensor as bt
import numpy as np
import pandas as pd
import torch

from zeus import __version__ as zeus_version
from zeus.base.miner import BaseMinerNeuron
from zeus.protocol import (
    PredictionSynapse,
    TimePredictionSynapse,
)
from zeus.commitment import ChallengeCommitment, commit_to_chain
from zeus.utils.compression import compress_prediction
from zeus.utils.hash import prediction_hash
from zeus.utils.time import to_timestamp
from zeus.validator.constants import CHALLENGE_REGISTRY, SHORT_CHALLENGE, CHALLENGE_HASHING_MAX_MINUTE

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior.
    In particular, you should replace the forward function with your own logic.

    Currently the base miner does a request to Herbie for predictions.
    You are encouraged to attempt to improve over this by changing the forward function.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info("Attaching forward functions to miner axon.")
        # Register both synapse types so the axon accepts TimePredictionSynapse.
        self.axon.attach(
            forward_fn=self._forward_unhashed_predictions,
            blacklist_fn=self._blacklist_time,
            priority_fn=self._priority_time
        )
        bt.logging.warning(f"Miner axon attached {self.axon}")
        self.pre_compute_predictions()
        
        # TODO(miner): Anything specific to your use case you can do here

        # Validators send requests to the miners for the forecast in the next 48hours in step of 1 hour every 6 hours starting at 00 (00, 06, 12, 18).
        # Because this is known and schedules, miners should precompute and save their forecast
        
        # The requests come in 3 stages:
        #   at 00, 06, 12, and 18 miners need to pass a hash of their predictions, this is the "commitment" stage
        #   1 hour after the commitment stage, some miners would be requested to send the actual forecast, unhashed
        #   Validators check if the unhashed predictions match the commited predictions, if a miner doesn't answer, or sends back a forecast different than the one which was hashed 40 minutes prior, it receives a penalty
        #   In a few days (7 days) when the ground truth becomes available, miners are requested to pass their forecast again, for the past dates. 
        #   the validators then check the commited hash with the forecast that the miners submitted and if it matches, the miners are scored. 
        # 
        # Therefore it is important that the miners keep their forecasts at least until the ground truth of the last time step of the forecasts becomes available and the miners are scored. 

        bt.logging.info("Precomputing predictions for 15 days and 48 hours")
        precomputed_forecast_15days = np.random.rand(24*15+1, 721, 1440).astype(np.float16)
        precomputed_forecast_49hours = np.random.rand(49, 721, 1440).astype(np.float16)
        self.compressed_forecast_15days = compress_prediction(precomputed_forecast_15days)
        self.compressed_forecast_49hours = compress_prediction(precomputed_forecast_49hours)
        bt.logging.info("Done precomputing prediction")

    def pre_compute_predictions(self):
        """
        This function precomputes and saves the predictions. 
        Given that the requests are at 00, 06, 12, 18 o'clock, 
        the predictions are precomputed one hour prior to request time. 

        This function return nothing
        """
        # TODO(miner): Anything specific to your use case you can do here 

    def on_challenge_block(self, challenge_time: datetime, subtensor=None):
        """Called when the chain reaches a challenge block (00/06/12/18 UTC).

        Builds one hash per (variable x time_window) challenge and commits
        them to chain as 2 x Raw128 fields (long + short).
        """
        sub = subtensor or self.subtensor
        bt.logging.info(f"Challenge triggered for {challenge_time}")
        self.pre_compute_predictions()

        commitment = ChallengeCommitment()
        hotkey = self.wallet.hotkey.ss58_address

        for state_key in sorted(CHALLENGE_REGISTRY):
            spec = CHALLENGE_REGISTRY[state_key]
            if (spec.start_offset, spec.end_offset) == SHORT_CHALLENGE:
                compressed = self.compressed_forecast_49hours
            else:
                compressed = self.compressed_forecast_15days

            h = prediction_hash(compressed, hotkey)
            commitment.hashes[state_key] = h
            bt.logging.info(f"  {state_key}: {h[:16]}...")

        bt.logging.info(f"Committing hashes to chain for {challenge_time}")
        commit_to_chain(
            sub,
            self.wallet,
            netuid=self.config.netuid,
            commitment=commitment,
        )
        bt.logging.success(f"Commitment uploaded for {challenge_time}")


    async def _forward_unhashed_predictions(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """Axon endpoint for reveal-phase (predictions) requests."""
        now = pd.Timestamp.now("UTC").replace(tzinfo=None)
        # Miners don't reveal outside of those hours as your forecast might be used for relay mining
        if (now.hour%6 == 0 and now.minute >= CHALLENGE_HASHING_MAX_MINUTE ) and to_timestamp(synapse.end_time) > now - pd.Timedelta(days = 4):
            return synapse
        
        bt.logging.warning(f"Prediction Request from validator hotkey: {synapse.dendrite.hotkey}")

        synapse.version = zeus_version
        if synapse.requested_hours == 49:
            synapse.predictions = base64.b64encode(self.compressed_forecast_49hours).decode("ascii")
        else:
            synapse.predictions = base64.b64encode(self.compressed_forecast_15days).decode("ascii")
        return synapse



    async def _blacklist_time(self, synapse: TimePredictionSynapse) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)



    async def _priority_time(self, synapse: TimePredictionSynapse) -> float:
        return await self.priority(synapse)

    async def blacklist(self, synapse: PredictionSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)
    
    async def priority(self, synapse: PredictionSynapse) -> float:
        return await self._priority(synapse)
    

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(30)
