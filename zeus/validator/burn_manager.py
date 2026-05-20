# The MIT License (MIT)
# Copyright © 2025 Ørpheus A.I.

import json
from pathlib import Path
from typing import Any, Callable, List, Optional

import bittensor as bt
import numpy as np
from zeus.validator.constants import BLOCKS_TO_REQUEST_BURN
from zeus.validator.time_till_next_epoch import time_till_next_epoch, current_tempo_bounds


def _apply_burn(raw_weights: np.ndarray, burn_pairs: list):
    """Set each (uid, value) in burn_pairs on raw_weights and scale the remaining weights proportionally."""
    if not burn_pairs:
        return
    total_burn = sum(v for _, v in burn_pairs)
    if total_burn > 1.0:
        bt.logging.error(
            f"[_apply_burn] Total burn ({total_burn:.4f}) exceeds 1.0 — burn not applied to avoid negative weights."
        )
        return
    burn_uids = [uid for uid, _ in burn_pairs]
    for uid, val in burn_pairs:
        raw_weights[uid] = val
    others_mask = np.ones(len(raw_weights), dtype=bool)
    for uid in burn_uids:
        others_mask[uid] = False
    others_sum = np.sum(raw_weights[others_mask])
    if others_sum > 0:
        raw_weights[others_mask] *= (1.0 - total_burn) / others_sum
    else:
        raw_weights[others_mask] = (1.0 - total_burn) / max(1, others_mask.sum())
    bt.logging.info(f"[set_weights] Burn applied: {burn_pairs}")

class BurnManager:
    """
    Owns burn interval state from the performance API and JSON persistence.
    """

    def __init__(
        self,
        *,
        storage_path: Path,
        netuid: int,
        performance_database_api: Any,
        on_fetch_failed: Optional[Callable[[], None]] = None,
    ):
        self._path = storage_path
        self._netuid = netuid
        self._performance_database_api = performance_database_api
        self._on_fetch_failed = on_fetch_failed

        self._intervals: List[Any] = []
        
        # Tracks the last epoch where we SUCCESSFULLY fetched data
        self._last_successful_fetch_epoch: Optional[int] = None
        
        # Tracks the last epoch where we FIRED the failure callback
        self._last_failed_callback_epoch: Optional[int] = None

    def load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path, "r") as f:
                    self._intervals = json.load(f)
         
                bt.logging.info(
                    f"[BurnManager] Loaded {len(self._intervals)} burn interval(s) from {self._path}"
                )
            else:
                self._intervals = []
                bt.logging.info(f"[BurnManager] No burn amounts on disk at {self._path}")
        except Exception as e:
            bt.logging.warning(f"[BurnManager] Could not load burn amounts: {e}")
            self._intervals = []

    def _save_unlocked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(self._intervals, f)
            tmp_path.replace(self._path)
        except Exception as e:
            bt.logging.warning(f"[BurnManager] Could not save burn amounts: {e}")

    def get_burn_entries_for_block(self, block: int) -> List[dict]:
        """Return a copy of the burns list for the interval covering block (empty if none)."""
        for interval in self._intervals:
            try:
                if interval["burn_block_start"] <= block <= interval["burn_block_end"]:
                    burns = interval.get("burns", [])
                    return [dict(b) for b in burns if isinstance(b, dict)]
            except (KeyError, TypeError):
                continue
        return []

    def refresh_if_needed(self, current_block: int, tempo: int) -> None:
        tte = time_till_next_epoch(current_block, self._netuid, tempo)
        
        # Calculate the exact block where the CURRENT epoch started.
        # This matches the math in WeightSetter._current_tempo_bounds()
        # next_epoch_start = current_block + tte + 1
        # current_epoch_start = next_epoch_start - (tempo + 1)
        current_epoch_start, _ = current_tempo_bounds(current_block, self._netuid, tempo)
        
        # Only fetch when we get close to the end of the current epoch
        if tte >= BLOCKS_TO_REQUEST_BURN:
            return
            
        # Lock 1: Have we already successfully fetched data during this epoch?
        if self._last_successful_fetch_epoch == current_epoch_start:
            bt.logging.debug(f"[BurnManager] Already successfully fetched data for epoch {current_epoch_start}")
            return
            
        # Pass the current epoch ID down to the fetch logic
        if self._fetch_and_persist(current_epoch_start):
            self._last_successful_fetch_epoch = current_epoch_start

    def _fetch_and_persist(self, current_epoch_start: int) -> bool:
        bt.logging.info(f"[BurnManager] Fetching burn amounts for epoch starting at {current_epoch_start}...")
        try:
            result = self._performance_database_api.fetch_burn_amounts_sync()
        except Exception as e:
            bt.logging.warning(f"[BurnManager] fetch burn amounts raised: {e}")
            result = None
        if result is not None:
            self._intervals = result
            self._save_unlocked()
            bt.logging.info(f"[BurnManager] Burn amounts updated: {len(result)} interval(s)")
            return True
        bt.logging.error(
            "[BurnManager] Failed to fetch burn amounts. Will retry next loop."
        )
        
        # Lock 2: Only trigger the failure callback ONCE per epoch
        if self._on_fetch_failed is not None and self._last_failed_callback_epoch != current_epoch_start:
            try:
                self._on_fetch_failed()
                self._last_failed_callback_epoch = current_epoch_start
                bt.logging.info(f"[BurnManager] on_fetch_failed callback triggered for epoch {current_epoch_start}")
            except Exception as e:
                bt.logging.warning(f"[BurnManager] on_fetch_failed callback error: {e}")
                
        return False
