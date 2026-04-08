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
import threading
from abc import abstractmethod
from traceback import format_exception
from typing import Dict, List, Set, Union

import bittensor as bt
import numpy as np

from zeus.base.dendrite import DendriteSettings, ZeusDendrite
from zeus.base.neuron import BaseNeuron
from zeus.base.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from zeus.utils.config import add_validator_args
from zeus.utils.results_state import ResultsState, load_state, save_state
from zeus.utils.uids import check_uid_availability, is_registered_after_release_zeus_v2 as is_after_zeus_v2_cutoff
from zeus.validator.constants import (
    CHALLENGE_REGISTRY,
    HASH_DENDRITE_SETTINGS,
    RANK_HISTORY_PRUNE_LEN,
    SHORT_CHALLENGE,
)
from zeus.validator.challenge_spec import ChallengeSpec
from zeus.validator.reward import compute_min_rank_weights


def _sum_weights_per_variable(registry: Dict[str, ChallengeSpec]) -> Dict[str, float]:
    """Total weight of all time-window challenges registered for each ERA5 variable."""
    weights: Dict[str, float] = {}

    for spec in registry.values():
        weights[spec.variable] = weights.get(spec.variable, 0.0) + spec.weight
    return weights


def _sum_available_weights_per_variable(
    available_keys: List[str], registry: Dict[str, ChallengeSpec]
) -> Dict[str, float]:
    """Total weight of the available (have rank_history) challenges for each variable."""
    weights: Dict[str, float] = {}
    for key in available_keys:
        var = registry[key].variable
        weights[var] = weights.get(var, 0.0) + registry[key].weight
    return weights


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
            self.BURN_UID = 56
            bt.logging.info(f"Burn functionality enabled. Using BURN_UID = {self.BURN_UID}.")

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()
        self.challenges = []

        # Hash dendrite (single, shared across all challenges)
        self.dendrite_hash = ZeusDendrite(
            wallet=self.wallet,
            settings=HASH_DENDRITE_SETTINGS,
        )

        # Prediction dendrites: one per unique DendriteSettings across all challenge windows
        unique_settings = {spec.prediction_dendrite_settings for spec in CHALLENGE_REGISTRY.values()}
        self.prediction_dendrites: Dict[DendriteSettings, ZeusDendrite] = {
            settings: ZeusDendrite(wallet=self.wallet, settings=settings)
            for settings in unique_settings
        }
        bt.logging.info(
            "Dendrites: hash=%s, prediction=%d unique settings",
            self.dendrite_hash, len(self.prediction_dendrites),
        )

        self.challenge_registry = CHALLENGE_REGISTRY

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        old_state_path = os.path.join(self.config.neuron.full_path, "state_v2.json")
        new_state_path = os.path.join(self.config.neuron.full_path, "state_v3.json")
        self.migrate_state(old_state_path, new_state_path)
        self.state_path = new_state_path
        
        self.state_per_challenge, loaded_step = load_state(
            self.state_path
        )
        if loaded_step is not None:
            self.step = loaded_step
        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

        # Init sync with the network. Updates the metagraph.
        self.sync_without_weights()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")
    
    def migrate_state(self, v2_path, v3_path):
        if os.path.exists(v3_path):
            bt.logging.info(
                f"Skip migration: {v3_path} already exists "
                f"(remove it first if you need to re-run v2 → v3 from {v2_path})."
            )
            return

        bt.logging.info(f"Looking for v2 state at: {v2_path}")
        
        if not os.path.exists(v2_path):
            bt.logging.error(f"Error: File {v2_path} does not exist.")
            return

        try:
            with open(v2_path, "r") as f:
                v2_data = json.load(f)
        except json.JSONDecodeError as e:
            bt.logging.error(f"Error: Could not parse {v2_path} as JSON: {e}")
            return

        step = v2_data.get("step")
        v2_variables = v2_data.get("variables", {})

        v3_variables: Dict[str, ResultsState] = {}

        for var_name, old_content in v2_variables.items():
            state_key = f"{var_name}@0_48"
            v3_variables[state_key] = ResultsState(
                name=state_key,
                rank_history=old_content.get("rank_history", {}).copy(),
                best_10_miners=old_content.get("best_10_miners", []).copy(),
            )
        
        save_state(v3_path, v3_variables, step)
        bt.logging.info(f"Successfully migrated {v2_path} to {v3_path}")

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

    async def concurrent_forward(self):
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

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

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

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

    def find_miners(self):
        miners_hotkeys = []
        miners_uids = []
        for uid,hotkey in zip(self.metagraph.uids, self.metagraph.hotkeys):
            
            available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit, self.metagraph.netuid)
            is_registered_after_v2 = self.is_registered_after_release_zeus_v2(uid)
            if available and is_registered_after_v2:
                miners_hotkeys.append(hotkey)
                miners_uids.append(uid)

        return miners_hotkeys, miners_uids

    def set_weights(self):
        """
        Sets the validator weights from rank_history: for each uid, average of last n ranks (inverted so lower rank -> higher weight), then normalize and emit.
        
        If burn_percent is configured, assigns that percentage to BURN_UID
        and distributes the remainder proportionally among other miners.
        """
        bt.logging.info("Setting weights")
        miners_hotkeys, miners_uids = self.find_miners()
        challenge_weights_list = []
        challenge_scaler = []

        available_keys = [
            challenge_name for challenge_name, state in self.state_per_challenge.items()
            if self.challenge_registry.get(challenge_name) is not None and state.rank_history
        ]

        if available_keys:
            total_weights_per_var = _sum_weights_per_variable(self.challenge_registry)
            available_weights_per_var = _sum_available_weights_per_variable(available_keys, self.challenge_registry)


            for state_key in available_keys:
                spec = self.challenge_registry[state_key]
                state = self.state_per_challenge[state_key]

                effective_weight = spec.weight * total_weights_per_var[spec.variable] / available_weights_per_var[spec.variable]
                bt.logging.info(
                    f"Calculating weights for {state_key} "
                    f"(effective_weight={effective_weight:.4f}, "
                    f"available_weight={available_weights_per_var[spec.variable]:.4f}/total_weight={total_weights_per_var[spec.variable]:.4f})"
                )

                max_len = max(len(state.rank_history[hotkey]) for hotkey in state.rank_history)
                
                if (spec.start_offset, spec.end_offset) == SHORT_CHALLENGE:
                    window_size = min(self.config.neuron.score_time_window_short, max_len)
                else:
                    window_size = min(self.config.neuron.score_time_window_long, max_len)

                weights, miners_metadata = compute_min_rank_weights(
                    metagraph_size=self.metagraph.n,
                    hotkeys=self.metagraph.hotkeys,
                    rank_history=state.rank_history,
                    uids=self.metagraph.uids,
                    window_size=window_size,
                    miners_hotkeys=miners_hotkeys,
                )
                state.best_10_miners = [m["hotkey"] for m in miners_metadata[:10]]
                challenge_weights_list.append(weights)
                challenge_scaler.append(effective_weight)

                self.performance_database_api.log_rank_aggregates(miners_metadata, state_key)

            save_state(
                self.state_path,
                self.state_per_challenge, 
                step=self.step,
            )

            raw_weights = np.average(challenge_weights_list, axis=0, weights=challenge_scaler)

        else: # no rank history for any variable
            bt.logging.warning("No rank history so rewarding responding miners first 7 days.")
            raw_weights = self.reward_responders( # of no responding miners were found, the weights would be all 0, but that handles below
                metagraph_size=self.metagraph.n, 
                uids=self.metagraph.uids, 
                hotkeys=self.metagraph.hotkeys
                )                 

        if np.isnan(raw_weights).any(): # This should not be possible, maybe we can delete it later
            bt.logging.warning("[set_weights] Computed weights contain NaN; zeroing. This should not happen, something is potentially going wrong")
            raw_weights = np.nan_to_num(raw_weights, nan=0.0)

        # Normalize scores to weights 
        weights_sum = np.sum(raw_weights)
        if weights_sum > 0:
            raw_weights = raw_weights / weights_sum
        else: # -> if no responding miners were found here is where the problem is fixed. 
            num_miners = len(miners_uids)
            if num_miners > 0:
                raw_weights = np.zeros_like(raw_weights)
                raw_weights[miners_uids] = 1.0
                raw_weights = raw_weights / num_miners
            else:
                raise ValueError("No miners found")

        # Apply burn logic if configured
        burn_percent = getattr(self.config.neuron, "burn_percent", 0.0)
        bt.logging.debug(f"[set_weights] Burning {burn_percent} percent ")
        if 0 < burn_percent < 1.0 and self.BURN_UID and self.BURN_UID < len(raw_weights):
            # Set burn UID to fixed percentage
            raw_weights[self.BURN_UID] = burn_percent
            
            # Scale all other weights proportionally to sum to (1 - burn_percent)
            others_mask = np.ones(len(raw_weights), dtype=bool)
            others_mask[self.BURN_UID] = False
            others_sum = np.sum(raw_weights[others_mask])
            
            if others_sum > 0:
                # Preserve relative ratios while scaling to target sum
                raw_weights[others_mask] *= (1.0 - burn_percent) / others_sum
            else:
                # Edge case: all others have zero weight, distribute uniformly
                raw_weights[others_mask] = (1.0 - burn_percent) / (len(raw_weights) - 1)
            
            bt.logging.info(f"Burn applied: UID {self.BURN_UID} set to {burn_percent:.2%}") 
        
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )


        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=True,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    @abstractmethod
    def get_responding_miners_hotkeys(self) -> Set[str]:
        """
        Returns a set of hotkeys that are responding to the validator.
        """
        pass

    def reward_responders(self, metagraph_size:int, uids: List[int], hotkeys: List[str]):
        weights = np.zeros(metagraph_size)
        responding_miners_hotkeys = self.get_responding_miners_hotkeys()

        for uid, hotkey in zip(uids, hotkeys):
            if hotkey in responding_miners_hotkeys and self.is_registered_after_release_zeus_v2(uid):
                weights[uid] = 1
        
        return weights 

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
        
        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        
        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Zero out all hotkeys that have been replaced but maybe they registered again in between
        hotkeys_to_prune = []
        for state in self.state_per_challenge.values():
            pruned_hotkeys = state.prune(self.hotkeys, RANK_HISTORY_PRUNE_LEN)
            if pruned_hotkeys:
                hotkeys_to_prune.extend(pruned_hotkeys)
        hotkeys_to_prune = list(set(hotkeys_to_prune))
        save_state(
            self.state_path,
            self.state_per_challenge,
            step=self.step,
        )

        self.prune_hotkeys(hotkeys_to_prune)

    @abstractmethod
    def prune_hotkeys(self, hotkeys: List[str]):
        """
        Prune data for hotkeys that got changed for their uid.
        """
        pass

    def save_state(self):
        """Persist variable state, step, and hotkeys to disk."""
        bt.logging.info("Saving validator state.")
        save_state(
            self.state_path,
            self.state_per_challenge,
            step=self.step,
        )

    def load_state(self):
        """Load variable state, step, and hotkeys from disk."""
        bt.logging.info("Loading validator state.")
        state_per_challenge, loaded_step = load_state(
            self.state_path
        )
        self.state_per_challenge = state_per_challenge
        if loaded_step is not None:
            self.step = loaded_step

    def update_scores(self, rewards: np.ndarray, hotkeys_list: List[str], state_key: str):
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
        state.insert_rank_history(rewards, hotkeys_list)
  
        bt.logging.debug(f"Appended ranks to rank_history for {state_key}: {rewards}")
        state.prune(self.metagraph.hotkeys, RANK_HISTORY_PRUNE_LEN)
        save_state(
            self.state_path,
            self.state_per_challenge,
            step=self.step,
        )

