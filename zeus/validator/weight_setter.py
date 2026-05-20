import copy
import threading
import time
from typing import Dict, List, Optional

import bittensor as bt
import numpy as np

from zeus.base.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from zeus.utils.results_state import load_rank_history_snapshot
from zeus.utils.uids import (
        find_miners,
        is_registered_after_release_zeus_v2 as is_after_zeus_v2_cutoff,
)
from zeus.validator.burn_manager import _apply_burn
from zeus.validator.challenge_spec import ChallengeSpec
from zeus.validator.constants import BLOCKS_TO_SET_WEIGHT, SHORT_CHALLENGE, BLOCKS_TO_REQUEST_BURN
from zeus.validator.reward import compute_avg_ranks, calculate_challenge_weights
from zeus.validator.time_till_next_epoch import time_till_next_epoch, current_tempo_bounds
from zeus.utils.misc import ttl_get_block

def _sum_weights_per_variable(registry: Dict[str, ChallengeSpec]) -> Dict[str, float]:
    # Moved unchanged from zeus/base/validator.py.
    weights: Dict[str, float] = {}
    for spec in registry.values():
        weights[spec.variable] = weights.get(spec.variable, 0.0) + spec.weight
    return weights


def _sum_available_weights_per_variable(
    available_keys: List[str],
    registry: Dict[str, ChallengeSpec],
) -> Dict[str, float]:
    # Moved unchanged from zeus/base/validator.py.
    weights: Dict[str, float] = {}
    for key in available_keys:
        var = registry[key].variable
        weights[var] = weights.get(var, 0.0) + registry[key].weight
    return weights


class WeightSetter:
    def __init__(
        self,
        *,
        config,
        wallet,
        challenge_registry,
        burn_uid,
        burn_manager,
        spec_version: int,
        get_responding_miners_hotkeys,
        blocks_to_wait_before_resetting_weights
    ):
        self.config = config
        self.wallet = wallet
        self.challenge_registry = challenge_registry # does not change after init
        self.burn_uid = burn_uid
        self.burn_manager = burn_manager
        self.spec_version = spec_version
        self.get_responding_miners_hotkeys = get_responding_miners_hotkeys
        self.blocks_to_wait_before_resetting_weights = blocks_to_wait_before_resetting_weights

        # Private websocket connection for this worker thread.
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.tempo = self.subtensor.tempo(self.config.netuid)

        self.thread: Optional[threading.Thread] = None
        self.should_exit = False

        self._last_update_block: Optional[int] = None
        self.sync()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self._refresh_last_update_from_metagraph()

    @property
    def block(self):
        return ttl_get_block(self)

    def _blocks_since_last_weight_update(self, current_block: int) -> int:
        return current_block - self._last_update_block

    def _is_set_weights_rate_limited(self, current_block: int) -> bool:
        return (
            self._blocks_since_last_weight_update(current_block)
            < self.blocks_to_wait_before_resetting_weights
        )

    def sync(self) -> None:
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            self.should_exit = True
            raise RuntimeError(
                f"Wallet hotkey is not registered on netuid {self.config.netuid}."
            )

        self.metagraph.sync(subtensor=self.subtensor)

    def _refresh_last_update_from_metagraph(self) -> None:
        self._last_update_block = int(self.metagraph.last_update[self.uid].item())

    def start(self):
        """Starts the background thread."""
        if self.thread is None or not self.thread.is_alive():
            bt.logging.info("[WeightSetter] Starting background thread...")
            self.should_exit = False
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Safely stops the thread."""
        self.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)

    def _run_loop(self):
        while not self.should_exit:
            try:

                self._refresh_burn_if_needed()

                
                self.set_weights_if_needed()

            except Exception as e:
                bt.logging.exception(f"[WeightSetter] Loop error: {e}")

            for _ in range(12):
                if self.should_exit:
                    break
                time.sleep(1)
    
    def _refresh_burn_if_needed(self):
        current_block = self.block
        tte = time_till_next_epoch(
            current_block,
            self.config.netuid,
            self.tempo,
        )
        if tte < BLOCKS_TO_REQUEST_BURN:
            if self.burn_manager is not None:
                self.burn_manager.refresh_if_needed(current_block, self.tempo)


    def set_weights_if_needed(self):
        current_block = self.block
        tte = time_till_next_epoch(
                    current_block,
                    self.config.netuid,
                    self.tempo,
                )
        if tte >= BLOCKS_TO_SET_WEIGHT:
            return
            
        epoch_start, epoch_end = current_tempo_bounds(current_block, self.config.netuid, self.tempo)

        if self._last_update_block >= epoch_start:
            bt.logging.info(
                f"[WeightSetter] Already successfully set weights for epoch starting on block {self._last_update_block} epoch start {epoch_start} epoch end {epoch_end}"
            )
            return
        # if we set the weights no need to spam the blockchain with sync 
        if self._is_set_weights_rate_limited(current_block):
            bt.logging.debug(
                f"[WeightSetter] Rate limited: {self._blocks_since_last_weight_update(current_block)} "
                f"< {self.blocks_to_wait_before_resetting_weights} blocks since last update "
                f"(last_update_block={self._last_update_block})"
            )
            return

        try:


            self.sync()
            self._refresh_last_update_from_metagraph()
            # extra sanity check to make sure we are not rate limited for setting weights if caching blocks we are off by one or 2 blocks
            if self._is_set_weights_rate_limited(self.block):
                bt.logging.info(
                    "[WeightSetter] Still rate limited after metagraph sync; skipping set_weights."
                )
                return
            
            successfully_set_weights = self.set_weights()
            if successfully_set_weights:
                self._last_update_block = self.block
            else:
                bt.logging.warning(
                    f"[WeightSetter] Failed to set weights for epoch starting on block {epoch_start}. Retrying next loop."
                )
        except Exception:
            # If an unexpected exception occurs, the loop catches it in _run_loop, 
            # and we naturally retry
            raise

    def set_weights(self) -> bool:
        """
        Sets the validator weights from rank_history: for each uid, average of last n ranks
        (inverted so lower rank -> higher weight), then normalize and emit.

        If burn_percent is configured, assigns that percentage to burn_uid
        and distributes the remainder proportionally among other miners.
        """
        
        metagraph = copy.deepcopy(self.metagraph)

        miners_hotkeys, miners_uids = find_miners(
            metagraph=metagraph,
            vpermit_tao_limit=self.config.neuron.vpermit_tao_limit,
            mainnet_uid=self.metagraph.netuid,
            current_block=self.block,
        )
        challenge_weights_list = []
        challenge_scaler = []

        state_keys = list(self.challenge_registry.keys())
        rank_snapshot = load_rank_history_snapshot(state_keys)

        available_keys = [
            challenge_name
            for challenge_name, history in rank_snapshot.items()
            if self.challenge_registry.get(challenge_name) is not None and history
        ]

        if available_keys:
            total_weights_per_var = _sum_weights_per_variable(self.challenge_registry)
            available_weights_per_var = _sum_available_weights_per_variable(available_keys, self.challenge_registry)


            for state_key in available_keys:
                spec = self.challenge_registry[state_key]
                rank_history_dict: Dict[str, List[float]] = rank_snapshot[state_key]

                effective_weight = spec.weight * total_weights_per_var[spec.variable] / available_weights_per_var[spec.variable]
                bt.logging.info(
                    f"Calculating weights for {state_key} "
                    f"(effective_weight={effective_weight:.4f}, "
                    f"available_weight={available_weights_per_var[spec.variable]:.4f}/total_weight={total_weights_per_var[spec.variable]:.4f})"
                )

                max_len = max(len(hotkey_rank_history) for hotkey_rank_history in rank_history_dict.values())
                
                if (spec.start_offset, spec.end_offset) == SHORT_CHALLENGE:
                    window_size = min(self.config.neuron.score_time_window_short, max_len)
                else:
                    window_size = min(self.config.neuron.score_time_window_long, max_len)

                miners_metadata = compute_avg_ranks(
                    hotkeys=metagraph.hotkeys,
                    rank_history=rank_history_dict,
                    uids=metagraph.uids,
                    window_size=window_size,
                    miners_hotkeys=miners_hotkeys,
                )
                weights = calculate_challenge_weights(
                    miners_metadata=miners_metadata,
                    metagraph_size=metagraph.n,
                )
    
                challenge_weights_list.append(weights)
                challenge_scaler.append(effective_weight)

                

            raw_weights = np.average(
                challenge_weights_list,
                axis=0,
                weights=challenge_scaler,
            )

        else:
            bt.logging.warning(
                "[WeightSetter] No rank history so rewarding responding miners first 7 days."
            )
            raw_weights = self.reward_responders(
                metagraph=metagraph,
                metagraph_size=metagraph.n,
                uids=metagraph.uids,
                hotkeys=metagraph.hotkeys,
                current_block=self.block,
            )

        if np.isnan(raw_weights).any():
            bt.logging.warning(
                "[WeightSetter] Computed weights contain NaN; zeroing. "
                "This should not happen, something is potentially going wrong"
            )
            raw_weights = np.nan_to_num(raw_weights, nan=0.0)

        # Normalize scores to weights 
        weights_sum = np.sum(raw_weights)
        if weights_sum > 0:
            raw_weights = raw_weights / weights_sum
        else:
            num_miners = len(miners_uids)
            if num_miners > 0:
                raw_weights = np.zeros_like(raw_weights)
                raw_weights[miners_uids] = 1.0
                raw_weights = raw_weights / num_miners
            else:
                bt.logging.error("[WeightSetter] No eligible miners found; skipping weight update.")
                return False

        # Apply burn: prefer per-epoch entries from the performance DB, fall back to burn_percent config.
        burn_entries: list = []
        if self.burn_manager is not None:
            burn_entries = self.burn_manager.get_burn_entries_for_block(self.block)
        burn_pairs = (
            [(b["uid"], b["value"]) for b in burn_entries if b.get("uid") < len(raw_weights) and 0 < b["value"] <= 1.0]
            if burn_entries
            else []
        )
        if not burn_pairs:
            burn_percent = self.config.neuron.burn_percent
            bt.logging.warning(
                f"[set_weights] No received epoch burn data for block {self.block}; using burn_percent={burn_percent}"
            )
            burn_pairs = (
                [(self.burn_uid, burn_percent)] if self.burn_uid and 0 < burn_percent <= 1.0 else []
            )
        _apply_burn(raw_weights, burn_pairs)

        processed_weight_uids, processed_weights = process_weights_for_netuid(
            uids=metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=metagraph,
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
            bt.logging.info("[WeightSetter] set_weights on chain successfully!")
            return True

        bt.logging.error("[WeightSetter] set_weights failed", msg)
        return False


    def reward_responders(
        self,
        *,
        metagraph,
        metagraph_size: int,
        uids: List[int],
        hotkeys: List[str],
        current_block: int,
    ):
        # Uses the deep-copied metagraph snapshot from set_weights for consistency.
        weights = np.zeros(metagraph_size)
        responding_miners_hotkeys = self.get_responding_miners_hotkeys()

        for uid, hotkey in zip(uids, hotkeys):
            reg_block = metagraph.block_at_registration[uid]
            if (
                hotkey in responding_miners_hotkeys
                and is_after_zeus_v2_cutoff(reg_block, current_block)
            ):
                weights[uid] = 1

        return weights