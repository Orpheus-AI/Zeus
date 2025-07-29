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
import os
from discord_webhook import DiscordWebhook, DiscordEmbed

import bittensor as bt
import wandb
from dotenv import load_dotenv

import zeus
from zeus.validator.uid_tracker import UIDTracker
from zeus.api.proxy import ValidatorProxy
from zeus.base.validator import BaseValidatorNeuron
from zeus.validator.forward import forward
from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.data.loaders.openmeteo import OpenMeteoLoader
from zeus.data.difficulty_loader import DifficultyLoader
from zeus.validator.database import ResponseDatabase
from zeus.validator.constants import (
    TESTNET_UID,
)


class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.load_state()

        load_dotenv(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../validator.env")
        )
        self.discord_hook = os.environ.get("DISCORD_WEBHOOK")

        self.uid_tracker = UIDTracker(self)
        self.validator_proxy = ValidatorProxy(self)

        bt.logging.info("Initialising data loaders...")
        self.cds_loader = Era5CDSLoader()
        self.open_meteo_loader = OpenMeteoLoader()
        bt.logging.info("Finished setting up data loaders.")

        self.database = ResponseDatabase(self.cds_loader)
        self.difficulty_loader = DifficultyLoader()
        self.init_wandb()

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
    
    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.validator_proxy.stop_server()


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
        if wandb.run and not wandb.run.offline:
            embed.add_embed_field(name="", value=f"[WANDB]({wandb.run.get_url()}/logs)", inline=False)
        webhook.add_embed(embed)
        webhook.execute()

    def init_wandb(self):
        if self.config.wandb.off:
            return

        run_name = f"validator-{self.uid}-{zeus.__version__}"
        self.config.run_name = run_name
        self.config.uid = self.uid
        self.config.hotkey = self.wallet.hotkey.ss58_address
        self.config.version = zeus.__version__
        self.config.type = self.neuron_type

        wandb_project = (
            self.config.wandb.testnet_project_name
            if self.config.netuid == TESTNET_UID
            else self.config.wandb.project_name
        )

        # Initialize the wandb run for the single project
        bt.logging.info(
            f"Initializing W&B run for '{self.config.wandb.entity}/{wandb_project}'"
        )
        try:
            run_id = wandb.init(
                name=run_name,
                project=wandb_project,
                entity=self.config.wandb.entity,
                config=self.config,
                dir=self.config.full_path,
                mode="offline" if self.config.wandb.offline else None
            ).id
        except wandb.UsageError as e:
            bt.logging.warning(e)
            bt.logging.warning("Did you run wandb login?")
            return

        # Sign the run to ensure it's from the correct hotkey
        signature = self.wallet.hotkey.sign(run_id.encode()).hex()
        self.config.signature = signature
        wandb.config.update(self.config, allow_val_change=True)

        bt.logging.success(f"Started wandb run {run_name}")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while not validator.should_exit:
            bt.logging.info(f"Validator running | uid {validator.uid} | {time.time()}")
            time.sleep(30)