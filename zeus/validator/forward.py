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

import time
from functools import partial
from typing import List

import bittensor as bt
from copy import deepcopy
from zeus.base.validator import BaseValidatorNeuron
from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.validator.hash_phase import run_all_hash_phases
from zeus.validator.constants import FORWARD_DELAY_SECONDS
from zeus.validator.prediction_phase import run_final_prediction_phase, run_initial_prediction_top_k_phases



async def forward(self: BaseValidatorNeuron):
    """
    The forward function is called by the validator. Commit and reveal phases are done in a single forward pass.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    start_forward = time.time()
    data_loader: Era5CDSLoader = self.cds_loader
    

    # If the validator has been deregistered, it will automatically stop the program
    self.check_registered()
    
    if self.time_scheduler.is_hash_commit_time():
        # Note : because the challenge starts at .floor('6h') of the time now, that means that at is_hash_commit_time and at is_query_best_time the same challenge would be returned
        self.challenges = data_loader.get_challenge_samples()
        bt.logging.info(f"Requesting hashes from all miners for {len(self.challenges)} challenges")
        await run_all_hash_phases(self, self.challenges)
    if not data_loader.is_ready():
        bt.logging.info("Data loader is not ready yet... Waiting until ERA5 data is downloaded.")
        time.sleep(max(0, FORWARD_DELAY_SECONDS - (time.time() - start_forward)))
        return
    if self.time_scheduler.is_query_best_time():
    # Note : because the challenge starts at .floor('6h') of the time now, that means that at is_hash_commit_time and at is_query_best_time the same challenge would be returned
        bt.logging.info("Requesting unhashed predictions from top k miners or random ones")
        previous_metagraph = deepcopy(self.metagraph)
        previous_hotkeys2uids = {hotkey: uid for uid, hotkey in enumerate(previous_metagraph.hotkeys)}
        self.resync_metagraph()
        await run_initial_prediction_top_k_phases(self, self.challenges, previous_hotkeys2uids)

    # based on the block and the readiness of the database, we decide if we should try to see if any challenges are ready for scoring 
    if self.database.should_score():
        bt.logging.info("Potentially scoring stored predictions for live ERA5 data.")
        self.resync_metagraph() 
        need_to_set_weights = await self.database.score_and_prune(score_func=partial(run_final_prediction_phase, self))
        if need_to_set_weights or self.should_set_weights():
            self.set_weights()

    time.sleep(max(0, FORWARD_DELAY_SECONDS - (time.time() - start_forward)))
        

            

