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
import argparse
import asyncio
import copy
import json
import os
import sys
import time
import threading
from abc import abstractmethod
from datetime import timedelta
from traceback import format_exception
from typing import Dict, List, Set, Union

import bittensor as bt
import numpy as np

from zeus.base.dendrite import DendriteSettings, ZeusDendrite
from zeus.base.neuron import BaseNeuron
from zeus.utils.config import add_validator_args
from zeus.utils.results_state import ResultsState, load_state, migrate_state_to_db, init_result_state_db, prune_rank_database
from zeus.utils.uids import is_registered_after_release_zeus_v2 as is_after_zeus_v2_cutoff
from zeus.validator.constants import (
    CHALLENGE_REGISTRY,
    BURN_UID
)



class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        if self.subtensor.network.lower() == "test":
            self.BURN_UID = None 
            bt.logging.warning("Burning is skipped on the test network (burn weight not set).")
        else:
            self.BURN_UID = BURN_UID
            bt.logging.info(f"Burn functionality enabled. Using BURN_UID = {self.BURN_UID}.")

        self.burn_manager = None  # Optional; concrete validator may set zeus.validator.burn_manager.BurnManager

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()
        self.challenges = []

  

        # Prediction dendrites: one per unique DendriteSettings across all challenge windows
        unique_settings = set()
        for spec in CHALLENGE_REGISTRY.values():
            unique_settings.add(spec.topk_dendrite_settings)
            unique_settings.add(spec.scoring_dendrite_settings)
            
        self.prediction_dendrites: Dict[DendriteSettings, ZeusDendrite] = {
            settings: ZeusDendrite(wallet=self.wallet, settings=settings)
            for settings in unique_settings
        }
        bt.logging.info(
            f"Dendrites: prediction={len(self.prediction_dendrites)} unique settings"
        )

        self.challenge_registry = CHALLENGE_REGISTRY

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        old_state_path = os.path.join(self.config.neuron.full_path, "state_v3.json")
        
        init_result_state_db()
        migrate_state_to_db(old_state_path)
        
        self.state_per_challenge = load_state()
        
        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None

        # Init sync with the network. Updates the metagraph.
        self.sync_without_weights()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")
    
    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the validator.
            - The validator's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.
        Raises:
            KeyboardInterrupt: If the validator is stopped by a manual interruption.
            Exception: For unforeseen errors during the validator's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync_without_weights()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                self.loop.run_until_complete(self.forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync_without_weights()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            sys.exit(0)

        # In case of unforeseen errors, the validator will log the error and restart.
        except Exception as err:
            err_message = ''.join(format_exception(type(err), err, err.__traceback__))
            self.should_exit = True
            self.on_error(err, err_message)

    def on_error(self, error: Exception, error_message: str):
        """
        Invoked when a validator encounters an exception during the run
        """
        bt.logging.error(f"Error during validation: {str(error)}")
        bt.logging.error(error_message)

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    
    @abstractmethod
    def get_responding_miners_hotkeys(self) -> Set[str]:
        """
        Returns a set of hotkeys that are responding to the validator.
        """
        pass



    def is_registered_after_release_zeus_v2(self, uid : int):
        """
        This functions gets a uid and hotkey and checks is the neuron was registered before the release of the new subnet version
        """  
        reg_block = self.metagraph.block_at_registration[uid]
        current_block = self.subtensor.block

        return is_after_zeus_v2_cutoff(reg_block, current_block)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return
        
        bt.logging.info("Metagraph updated, re-syncing hotkeys")
        
        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Zero out all hotkeys that have been replaced but maybe they registered again in between
        hotkeys_to_prune = prune_rank_database(self.hotkeys)
        hotkeys_to_prune = list(set(hotkeys_to_prune))
        
        self.prune_hotkeys(hotkeys_to_prune)

    @abstractmethod
    def prune_hotkeys(self, hotkeys: List[str]):
        """
        Prune data for hotkeys that got changed for their uid.
        """
        pass

    

    def update_scores(self, rewards: np.ndarray, hotkeys_list: List[str], miner_penalty_bool_list: List[bool], state_key: str, challenge_enddate: float):
        """Appends rank (reward) to rank_history for each hotkey. rewards are ranks from set_rewards (min RMSE -> 1).
        Attribution is by hotkey so it remains correct after metagraph resync or delayed scoring."""

        if state_key not in self.state_per_challenge:
            if state_key not in self.challenge_registry:
                raise ValueError(f"state_key {state_key} not found in challenge registry.")
            self.state_per_challenge[state_key] = ResultsState(name=state_key)

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            rewards = np.nan_to_num(rewards, nan=0)

        rewards = np.asarray(rewards)

        if rewards.size == 0 or len(hotkeys_list) == 0:
            bt.logging.warning(
                "Either rewards or hotkeys is empty. No updates will be performed."
            )
            return

        if rewards.size != len(hotkeys_list):
            raise ValueError(
                f"Shape mismatch: rewards length {rewards.size} does not match hotkeys length {len(hotkeys_list)}"
            )

        
        state = self.state_per_challenge[state_key]
        state.insert_rank_history(rewards, hotkeys_list, miner_penalty_bool_list, challenge_enddate)
        self.update_best_miners_for_state(state_key)
  
        bt.logging.debug(f"Appended ranks to rank_history for {state_key}: {rewards}")
        prune_rank_database(self.metagraph.hotkeys)
