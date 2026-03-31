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
from typing import List, Set

import bittensor as bt
from discord_webhook import DiscordEmbed, DiscordWebhook
from dotenv import load_dotenv

import zeus
from zeus.base.validator import BaseValidatorNeuron
from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.utils.schedule_time import Scheduler
from zeus.validator.constants import (
    TESTNET_UID,
    BEST_FORECASTS_DIRECTORY, 
    PERFORMANCE_DATABASE_URL
)
from zeus.validator.forward import forward
from zeus.validator.storage import OptimizedWeatherStorage
from zeus.validator.uid_tracker import UIDTracker
from zeus.validator.performance_database_connection import PerformanceDatabaseConnection

class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.load_state()

        load_dotenv(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../validator.env")
        )
        self.discord_hook = os.environ.get("DISCORD_WEBHOOK")

        self.uid_tracker = UIDTracker(self)
        self.performance_database_api = PerformanceDatabaseConnection(
            wallet=self.wallet,
            api_url = PERFORMANCE_DATABASE_URL
        )
        self.time_scheduler = Scheduler()
        self.latest_good_miners_per_challenge: dict[str, List[int]] = None
    
        bt.logging.info("Initialising data loaders...")
        self.cds_loader = Era5CDSLoader()
        bt.logging.info("Finished setting up data loaders.")

        self.database = OptimizedWeatherStorage(self.cds_loader)
        self.best_predictions_path = BEST_FORECASTS_DIRECTORY

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
        if hasattr(self, 'performance_database_api'):
            self.performance_database_api.close()

    def on_error(self, error: Exception, error_message: str):
        super().on_error(error, error_message)

        if not self.discord_hook:
            return
        
        webhook = DiscordWebhook(
            url=self.discord_hook, 
            avatar_url="https://raw.githubusercontent.com/Orpheus-AI/Zeus/refs/heads/v1/static/zeus-icon.png",
            username="Zeus Subnet Bot",
            content=f"Your validator had an error -- see below!",
            timeout=5,
        )
        embed = DiscordEmbed(title=repr(error), description=error_message)
        embed.set_timestamp()
        webhook.add_embed(embed)
        webhook.execute()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while not validator.should_exit:
            bt.logging.info(f"Validator running | uid {validator.uid} | {time.time()}")
            time.sleep(30)