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


import os
import time
from typing import List, Set, Dict

import bittensor as bt
from discord_webhook import DiscordEmbed, DiscordWebhook
from dotenv import load_dotenv

from zeus.base.validator import BaseValidatorNeuron
from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.utils.schedule_time import Scheduler
from zeus.validator.burn_manager import BurnManager
from zeus.validator.constants import (
    BURN_AMOUNTS_JSON_PATH,
    BEST_FORECASTS_DIRECTORY,
    PERFORMANCE_DATABASE_URL,
)
from zeus.validator.forward import forward
from zeus.validator.storage import OptimizedWeatherStorage
from zeus.validator.uid_tracker import UIDTracker
from zeus.validator.performance_database_connection import PerformanceDatabaseConnection
from zeus.validator.weight_setter import WeightSetter
from zeus.validator.reward import compute_avg_ranks
from zeus.utils.uids import find_miners
from zeus.validator.constants import SHORT_CHALLENGE

class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        load_dotenv(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../validator.env")
        )
        self.discord_hook = os.environ.get("DISCORD_WEBHOOK")

        self.uid_tracker = UIDTracker(self)
        self.performance_database_api = PerformanceDatabaseConnection(
            wallet=self.wallet,
            api_url=PERFORMANCE_DATABASE_URL,
        )
        self.time_scheduler = Scheduler()
        self.latest_good_miners_per_challenge: dict[str, List[int]] = None

        self.burn_manager = BurnManager(
            storage_path=BURN_AMOUNTS_JSON_PATH,
            netuid=self.config.netuid,
            performance_database_api=self.performance_database_api,
            on_fetch_failed=self._on_burn_fetch_failed,
        )
        self.burn_manager.load()

        bt.logging.info("Initialising data loaders...")
        self.cds_loader = Era5CDSLoader()
        bt.logging.info("Finished setting up data loaders.")

        self.database = OptimizedWeatherStorage(self.cds_loader)
        self.best_predictions_path = BEST_FORECASTS_DIRECTORY

        hyperparams = self.subtensor.get_subnet_hyperparameters(self.config.netuid)

        # 4. Extract the limit values
        blocks_to_wait_before_resetting_weights = hyperparams.weights_rate_limit
        # none on testnet!
        blocks_to_wait_before_resetting_weights = 100 if blocks_to_wait_before_resetting_weights is None else blocks_to_wait_before_resetting_weights
        bt.logging.info(f"Blocks to wait before resetting weights: {blocks_to_wait_before_resetting_weights}")

        self.weight_setter = WeightSetter(
            config=self.config,
            wallet=bt.wallet(config=self.config),
            challenge_registry=self.challenge_registry,
            burn_uid=self.BURN_UID,
            burn_manager=self.burn_manager,
            spec_version=self.spec_version,
            get_responding_miners_hotkeys=self.get_responding_miners_hotkeys,
            blocks_to_wait_before_resetting_weights = blocks_to_wait_before_resetting_weights
        )
        self.weight_setter.start()

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)
    
    def prune_hotkeys(self, hotkeys):
        if self.is_running: # make sure init is finalised
            self.database.prune_hotkeys(hotkeys)

    def get_responding_miners_hotkeys(self) -> Set[str]:
        return self.database.get_responding_miners_hotkeys()
    
    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self, 'weight_setter'):
            self.weight_setter.stop()
        if hasattr(self, 'performance_database_api'):
            self.performance_database_api.close()

    def _on_burn_fetch_failed(self):
        self._send_discord_notification(
            content="Failed to fetch burn amounts from the performance database!",
            title="Burn amounts fetch failed",
            description="All retries to /get_burn_amounts were exhausted. Falling back to default burn values.",
        )

    def _send_discord_notification(self, content: str, title: str, description: str):
        if not self.discord_hook:
            return
        try:
            webhook = DiscordWebhook(
                url=self.discord_hook,
                avatar_url="https://raw.githubusercontent.com/Orpheus-AI/Zeus/refs/heads/v1/static/zeus-icon.png",
                username="Zeus Subnet Bot",
                content=content,
                timeout=5,
            )
            embed = DiscordEmbed(title=title, description=description)
            embed.set_timestamp()
            webhook.add_embed(embed)
            webhook.execute()
        except Exception as e:
            bt.logging.warning(f"[Validator] Could not send Discord notification: {e}")

    def on_error(self, error: Exception, error_message: str):
        super().on_error(error, error_message)
        self._send_discord_notification(
            content="Your validator had an error -- see below!",
            title=repr(error),
            description=error_message,
        )
    
    def update_best_miners_for_state(self, state_key: str):
        """Refresh the ordered top miners for one challenge after rank history changes."""
        # similar to set_weights in weight_setter.py but we need to write only in the forward function
        if state_key not in self.state_per_challenge:
            bt.logging.warning(f"[update_best_miners_for_state] State key {state_key} not found in state_per_challenge.")
            return

        state = self.state_per_challenge[state_key]


        spec = self.challenge_registry[state_key]
 

        miners_hotkeys, _ = find_miners(
            self.metagraph,
            self.config.neuron.vpermit_tao_limit,
            self.metagraph.netuid,
            current_block=self.block,
        )

        rank_history_dict:Dict[str, List[float]] = state.rank_history
        if not rank_history_dict:
            bt.logging.warning(f"[update_best_miners_for_state] Rank history is empty for {state_key}.")
            return
     
        max_len = max(len(hotkey_rank_history) for hotkey_rank_history in rank_history_dict.values())
        
        if (spec.start_offset, spec.end_offset) == SHORT_CHALLENGE:
            window_size = min(self.config.neuron.score_time_window_short, max_len)
        else:
            window_size = min(self.config.neuron.score_time_window_long, max_len)
     
        miners_metadata = compute_avg_ranks(
            hotkeys=self.metagraph.hotkeys,
            rank_history=rank_history_dict,
            uids=self.metagraph.uids,
            window_size=window_size,
            miners_hotkeys=miners_hotkeys,
        )
        state.best_10_miners = [m["hotkey"] for m in miners_metadata[:10]]
        self.performance_database_api.log_rank_aggregates(miners_metadata, state_key)
        

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while not validator.should_exit:
            bt.logging.info(f"Validator running | uid {validator.uid} | {time.time()}")
            time.sleep(30)
