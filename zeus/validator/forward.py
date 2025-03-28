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
from functools import partial
import time
import bittensor as bt
import wandb
import numpy as np
import torch

from zeus.data.sample import Era5Sample
from zeus.data.era5.era5_cds import Era5CDSLoader
from zeus.utils.misc import split_list
from zeus.utils.time import timestamp_to_str
from zeus.utils.coordinates import bbox_to_str
from zeus.validator.reward import set_rewards
from zeus.validator.miner_data import MinerData
from zeus.utils.uids import get_random_uids
from zeus.validator.constants import FORWARD_DELAY_SECONDS, MAINNET_UID


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # based on the block, we decide if we should score old stored predictions.
    if self.database.should_score(self.block):
        bt.logging.info(f"Scoring all stored predictions for live ERA5 data.")
        self.database.score_and_prune(score_func=partial(complete_challenge, self))
        return

    data_loader: Era5CDSLoader = self.cds_loader
    if not data_loader.is_ready():
        bt.logging.info(
            "Data loader is not ready yet... Waiting until ERA5 data is downloaded."
        )
        time.sleep(10)  # Don't need to spam above message
        return

    bt.logging.info(f"Sampling data...")
    sample = data_loader.get_sample()
    bt.logging.success(
        f"Data sampled with bounding box {bbox_to_str(sample.get_bbox())}"
    )
    bt.logging.success(
        f"Data sampled starts from {timestamp_to_str(sample.start_timestamp)} | Asked to predict {sample.predict_hours} hours ahead."
    )

    miner_uids = get_random_uids(
        self.metagraph,
        self.config.neuron.sample_size,
        self.config.neuron.vpermit_tao_limit,
        MAINNET_UID,
    )

    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    miner_hotkeys: List[str] = list([axon.hotkey for axon in axons])

    bt.logging.info(f"Querying {len(miner_uids)} miners..")
    start = time.time()
    responses = await self.dendrite(
        axons=axons,
        synapse=sample.get_synapse(),
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )

    bt.logging.success(f"Responses received in {time.time() - start}s")

    # Create a dummy output with the same shape as the sample's prediction grid to check for penalties
    sample.output_data = torch.zeros(
        sample.predict_hours, sample.x_grid.shape[0], sample.x_grid.shape[1]
    )
    miners_data = get_scored_miners(self, sample, miner_hotkeys, responses)
    # Identify miners who should receive a penalty
    good_miners, bad_miners = split_list(miners_data, lambda m: m.penalty == 0.0)

    if len(bad_miners) > 0:
        uids = [miner.uid for miner in bad_miners]
        bt.logging.success(f"Punishing miners that did not respond immediately: {uids}")
        self.update_scores(
            [miner.reward for miner in bad_miners],
            uids,
        )
        do_wandb_logging(self, sample, bad_miners)

    if len(good_miners) > 0:
         # store non-penalty miners for proxy
        self.last_responding_miner_uids = [m.uid for m in good_miners]

        bt.logging.success("Storing challenge and sensible miner responses in SQLite database")
        self.database.insert(
            sample,
            [miner.hotkey for miner in good_miners],
            [miner.prediction for miner in good_miners],
        )

    # Introduce a delay to prevent spamming requests - and so miners should stay under free tier API request limit
    time.sleep(FORWARD_DELAY_SECONDS - (time.time() - start))


def complete_challenge(
    self,
    sample: Era5Sample,
    hotkeys: List[str],
    predictions: List[torch.Tensor],
) -> Optional[List[MinerData]]:
    """
    Complete a challenge by reward all miners. Based on hotkeys to also work for delayed rewarding.
    Note that non-responding miners (which get a penalty) have already been excluded.
    """

    miners_data = get_scored_miners(self, sample, hotkeys, predictions)

    self.update_scores(
        [miner.reward for miner in miners_data],
        [miner.uid for miner in miners_data],
    )

    for miner in miners_data:
        bt.logging.success(
            f"UID: {miner.uid} | Predicted shape: {miner.prediction.shape} | Reward: {miner.reward} | Penalty: {miner.penalty}"
        )
    do_wandb_logging(self, sample, miners_data)


def get_scored_miners(
    self,
    sample: Era5Sample,
    hotkeys: List[str],
    predictions: List[torch.Tensor],
) -> List[MinerData]:
    """
    Convert input to MinerData and calculate (and populate) their score/metrics fields.
    Return a list of MinerData
    """
    lookup = {axon.hotkey: uid for uid, axon in enumerate(self.metagraph.axons)}

    # Make miner data for each miner that is still alive
    miners_data = []
    for hotkey, prediction in zip(hotkeys, predictions):
        uid = lookup.get(hotkey, None)
        if uid is not None:
            miners_data.append(MinerData(uid=uid, hotkey=hotkey, prediction=prediction))

    # score and reward just those miners
    return set_rewards(
        output_data=sample.output_data,
        miners_data=miners_data,
        difficulty_grid=self.difficulty_loader.get_difficulty_grid(sample),
    )


def do_wandb_logging(self, challenge: Era5Sample, miners_data: List[MinerData]):
    if self.config.wandb.off:
        return
    
    for miner in miners_data:
        wandb.log(
            {f"miner_{miner.uid}_{key}": val for key, val in miner.metrics.items()},
            commit=False,  # All logging should be the same commit
        )

    wandb.log(
        {
            "start_timestamp": challenge.start_timestamp,
            "end_timestamp": challenge.end_timestamp,
            "predict_hours": challenge.predict_hours,
            "lat_lon_bbox": challenge.get_bbox(),
        },
    )

