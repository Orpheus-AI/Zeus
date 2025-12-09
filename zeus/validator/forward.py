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

from typing import List, Optional, Union, Dict
from functools import partial
import time
import random

import bittensor as bt
import wandb
import numpy as np
import torch

from zeus.data.base.loader import BaseDataLoader
from zeus.data.base.predictor import BasePredictor
from zeus.data.base.sample import BaseSample
from zeus.utils.misc import split_list
from zeus.utils.time import timestamp_to_str
from zeus.validator.reward import set_rewards, set_penalties, rmse
from zeus.validator.miner_data import MinerData
from zeus.utils.logging import maybe_reset_wandb
from zeus.base.validator import BaseValidatorNeuron
from zeus.protocol import PredictionSynapse
from zeus.validator.uid_tracker import UIDTracker
from zeus.validator.database import ResponseDatabase
from zeus.validator.constants import (
    MechanismType, 
    FORWARD_DELAY_SECONDS, 
    MECHANISM_PROBABILITIES
)

def sample_mechanism(self: BaseValidatorNeuron) -> MechanismType:
    """
    Sample a mechanism based on the probabilities.
    Only considers mechanisms where the dataloader is ready to sample.

    Returns:
        MechanismType: The sampled mechanism.
    """
    choices = [
        mtype for mtype, loader in self.data_loaders.items() 
        if loader.is_ready()
    ]
    weights = [MECHANISM_PROBABILITIES[mtype] for mtype in choices]
    return random.choices(choices, weights=weights)[0]

async def forward(self: BaseValidatorNeuron):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    start_forward = time.time()
    # Enable type checking since we cannot import actual neuron implementation here
    data_loaders: Dict[MechanismType, BaseDataLoader] = self.data_loaders
    baseline_loaders: Dict[MechanismType, BasePredictor] = self.baseline_loaders
    database: ResponseDatabase = self.database
    uid_tracker: UIDTracker = self.uid_tracker

    # based on the block, we decide if we should score old stored predictions.
    for mechanism, loader in data_loaders.items():
        if database.should_score(self.block, dataloader=loader):
            bt.logging.info(f"Potentially scoring stored predictions for {mechanism.name} data.")
            database.score_and_prune(loader, score_func=partial(complete_challenge, self))
    
    # every n forwards, we set preferences
    if self.preference_manager.should_query(self.block):
        bt.logging.info("Querying miner mechanism preference...")

        valid_uids, invalid_uids = await self.preference_manager.query_preferences(self.dendrite, self.metagraph)
        if invalid_uids:
            bt.logging.warning(f"Failed to fetch valid preferences for UIDS: {invalid_uids}. These have not been updated.")
        bt.logging.info(f"These miners have specified their desired mechanism! {valid_uids}")

    mechanism = sample_mechanism(self)
    bt.logging.info(f"Sampling challenge for mechanism: {mechanism.name}")
    
    sample = data_loaders[mechanism].get_sample()
    bt.logging.success(
        f"Data sampled for variable {sample.variable}"
    )
    bt.logging.success(
        f"Data sampled starts from {timestamp_to_str(sample.start_timestamp)} | Asked to predict {sample.hours_to_predict} hours ahead."
    )

    bt.logging.info("Fetching baseline(s)!")
    # get the baseline data, which we also store and check against
    sample.baseline = baseline_loaders[mechanism].get_forecast(sample)
    if not torch.isfinite(sample.baseline).all():
        bt.logging.warning("Baseline contains NaN or Inf values, skipping this sample.")
        return

    if mechanism == MechanismType.ERA5:
        # Not used for scoring, but also logged to W&B for comparisons of miner quality
        sample.ifs_hres_baseline = baseline_loaders[mechanism].get_forecast(sample, model="ecmwf_ifs")
  
    miner_uids = uid_tracker.get_random_uids(
        k = self.config.neuron.sample_size,
        mechanism=mechanism,
        tries = 3
    )

    if len(miner_uids) == 0:
        bt.logging.warning("No miners have this preferred challenge yet -- Sleeping!")
        time.sleep(10)
        return

    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    bt.logging.info(f"Querying {len(miner_uids)} miners..")
    responses: List[PredictionSynapse] = await self.dendrite(
        axons=axons,
        synapse=sample.get_synapse(),
        deserialize=False,
        timeout=self.config.neuron.timeout,
    )

    resp_times = [float(r.dendrite.process_time) for r in responses if r.dendrite.process_time]
    if resp_times:
        bt.logging.success(f"Received {len(resp_times)} responses with average response time of {np.mean(resp_times):.2f}s")

    miners_data = parse_miner_inputs(
        self, 
        sample=sample, 
        hotkeys=[axon.hotkey for axon in axons], 
        predictions=[r.deserialize() for r in responses],
        response_times=[r.dendrite.process_time for r in responses]    
    )
    # Identify miners who should receive a penalty
    good_miners, bad_miners = split_list(miners_data, lambda m: not m.shape_penalty)

    # penalise 
    if len(bad_miners) > 0:
        uids = [miner.uid for miner in bad_miners]
        uid_tracker.mark_finished(uids, mechanism=mechanism, good=False)
        bt.logging.success(f"Punishing miners that got a penalty: {uids}")
        self.update_scores(
            [miner.score for miner in bad_miners],
            uids,
            mechanism=mechanism,
        )
        do_wandb_logging(self, sample, bad_miners)

    if len(good_miners) > 0:
        uids = [m.uid for m in good_miners]
        # store non-penalty miners for proxy
        uid_tracker.mark_finished(uids, mechanism=mechanism, good=True)
      
        bt.logging.success(f"Storing challenge and sensible miner responses in SQLite database: {uids}")
        database.insert(sample, miners_data=good_miners)

    # prevent W&B logs from becoming massive
    maybe_reset_wandb(self)
    # Introduce a delay to prevent spamming requests
    time.sleep(max(0, FORWARD_DELAY_SECONDS - (time.time() - start_forward)))


def parse_miner_inputs(
    self: BaseValidatorNeuron,
    sample: BaseSample,
    hotkeys: List[str],
    predictions: List[torch.Tensor],
    response_times: List[Optional[Union[str, float]]]
) -> List[MinerData]:
    """
    Convert input to MinerData and calculate (and populate) their penalty fields.
    Return a list of MinerData
    """
    lookup = {axon.hotkey: uid for uid, axon in enumerate(self.metagraph.axons)}

    # Make miner data for each miner that is still alive
    miners_data = []
    for hotkey, prediction, response_time in zip(hotkeys, predictions, response_times):
        uid = lookup.get(hotkey, None)
        if uid is not None:
            miners_data.append(
                MinerData(
                    uid=uid, 
                    hotkey=hotkey, 
                    response_time=float(response_time or self.config.neuron.timeout), 
                    prediction=prediction
                )
            )

    # pre-calculate penalities since we need those to filter
    return set_penalties(
        correct_shape=sample.baseline.shape,
        miners_data=miners_data
    )


def complete_challenge(
    self: BaseValidatorNeuron,
    sample: BaseSample,
    hotkeys: List[str],
    predictions: List[torch.Tensor],
    response_times: List[float]
):
    """
    Complete a challenge by reward all miners. Based on hotkeys to also work for delayed rewarding.
    Note that non-responding miners (which get a penalty) have already been excluded.
    """
    
    miners_data = parse_miner_inputs(self, sample, hotkeys, predictions, response_times)
    miners_data = set_rewards(
        output_data=sample.output_data, 
        miners_data=miners_data, 
        baseline_data=sample.baseline,
        difficulty_grid=self.difficulty_loader.get_difficulty_grid(sample),
        challenge_age=self.data_loaders[sample.mechanism].get_relative_age(sample),
    )

    self.update_scores(
        [miner.score for miner in miners_data],
        [miner.uid for miner in miners_data],
        mechanism=sample.mechanism,
    )
    
    bt.logging.success(f"Scored stored {sample.mechanism.name} challenge for uids: {[miner.uid for miner in miners_data]}")
    for miner in miners_data:
        bt.logging.debug(
            f"UID: {miner.uid} | Predicted shape: {miner.prediction.shape} | Reward: {miner.score} | Penalty: {miner.shape_penalty}"
        )
    do_wandb_logging(self, sample, miners_data)


def do_wandb_logging(
        self, 
        challenge: BaseSample, 
        miners_data: List[MinerData], 
    ):
    """
    Log the challenge and miners data to W&B.
    """
    if self.config.wandb.off:
        return
    
    for miner in miners_data:
        wandb.log(
            {f"miner_{challenge.variable}_{miner.uid}_{key}": val for key, val in miner.metrics.items()},
            commit=False,  # All logging should be the same commit
        )

    uid_to_hotkey = {miner.uid: miner.hotkey for miner in miners_data}
    if challenge.output_data is not None:
        wandb.log({"baseline_rmse": rmse(challenge.output_data, challenge.baseline, default=np.nan)}, commit=False)
    
        if challenge.mechanism == MechanismType.ERA5:
            wandb.log({"ifs_hres_rmse": rmse(challenge.output_data, challenge.ifs_hres_baseline, default=np.nan)}, commit=False)
    wandb.log(
        {
            "mechanism": challenge.mechanism.name,
            "query_timestamp": challenge.inserted_at,
            "variable": challenge.variable,
            "start_timestamp": challenge.start_timestamp,
            "end_timestamp": challenge.end_timestamp,
            "predict_hours": challenge.hours_to_predict,
            "lat_lon_bbox": challenge.get_bbox(),
            "uid_to_hotkey": uid_to_hotkey,
        },
    )