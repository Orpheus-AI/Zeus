import math
import os
import random
import time
from typing import Callable, List, Optional, Tuple

import bittensor as bt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from zeus.base.dendrite import DendriteSettings, ZeusDendrite
from zeus.base.validator import BaseValidatorNeuron
from zeus.data.sample import Era5Sample
from zeus.protocol import TimePredictionSynapse
from zeus.utils.compression import decompress_prediction
from zeus.utils.time import to_timestamp
from zeus.validator.miner_data import MinerData
from zeus.validator.responses_processing import (
    _build_bad_miners_data,
    _verify_hashes,
    create_compressed_predictions,
)
from zeus.validator.reward import calculate_rmses, complete_challenge
from zeus.validator.storage import save_best_miner_prediction

def filter_eligible_miners_for_scoring(
    validator: BaseValidatorNeuron,
    current_challenge_all_miner_hotkeys: List[str],
    miner_uids_list: List[int],
    hashes_list: List[str],
    list_is_good: List[bool],
) -> Tuple[List[int], List[str], List[bool], List[str]]:
    """
    From challenge info, compute axons and uids to query (miners in challenge
    and registered before the challenge query timestamp). Returns
    (axons_to_query_all, uids_to_query_all, commitment_dict).
    """


    # create lists of MinerData objects for the miners that are in the challenge and registered before the challenge query timestamp
    miners_uids = []
    miners_hashes = []
    miners_is_good = []
    miners_hotkeys = []
    # for uid in range(len(validator.metagraph.axons)):
    for challenge_hotkey, miner_uid, is_good, hash in zip(current_challenge_all_miner_hotkeys, miner_uids_list, list_is_good, hashes_list):

        current_metagraph_hotkey = validator.metagraph.axons[miner_uid].hotkey

        # no need to query a new miner 
        if current_metagraph_hotkey != challenge_hotkey:
            continue
        
        is_registered_after_v2 = validator.is_registered_after_release_zeus_v2(miner_uid)
        if not is_registered_after_v2:
            continue

        miners_uids.append(miner_uid)
        miners_hashes.append(hash)
        miners_is_good.append(is_good)
        miners_hotkeys.append(challenge_hotkey)

    return miners_uids, miners_hashes, miners_is_good, miners_hotkeys



def _get_prediction_dendrite_and_settings(
    self, sample: Era5Sample
) -> Tuple[ZeusDendrite, DendriteSettings]:
    spec = self.challenge_registry.get(sample.state_key)
    if spec is None:
        raise ValueError(f"No ChallengeSpec for state_key={sample.state_key}")
    settings = spec.prediction_dendrite_settings
    return self.prediction_dendrites[settings], settings


async def fetch_predictions_and_verify_hashes(self, sample: Era5Sample, hashes_list: List[str], axons_to_query: List[bt.Axon]) -> List[Optional[bytes]]:
    """
    Handles a single batch of miners. Variables here are cleared from memory 
    once the function returns to run_single_hash_challenge.
    """

    dendrite, pred_settings = _get_prediction_dendrite_and_settings(self, sample)

    start_time = time.time()
    responses: List[TimePredictionSynapse] = await dendrite(
        axons=axons_to_query,
        synapse=sample.build_synapse(TimePredictionSynapse),
        deserialize=False,
        timeout=pred_settings.forward_timeout,
    )
    end_time = time.time()
    bt.logging.success(f"[fetch_predictions_and_verify_hashes] Received {len(responses)} responses in {end_time - start_time} seconds")


    compressed_predictions = create_compressed_predictions(responses)
 
    compressed_predictions = _verify_hashes(axons_to_query, compressed_predictions, hashes_list)
    return compressed_predictions


async def run_final_prediction_phase(self, sample, current_challenge_all_miner_hotkeys, miner_uids, hashes, is_good_list):
    """Run final prediction phase for scoring, verifying hashes and calculating metrics.
    
    Args:
        self: BaseValidatorNeuron instance
        sample: Era5Sample challenge to process
        current_challenge_all_miner_hotkeys: List of all miner hotkeys in the challenge
        miner_uids: List of miner UIDs
        hashes: List of prediction hashes from miners
        is_good: List of boolean flags indicating which miners are good
        
    Returns:
        List of all MinerData objects (good, bad, and non-committed miners)
    """
    # we want to score only those miners that were registered after the release of v2 and were not deregistered
    miners_uids, miners_hashes, miners_is_good, _ = filter_eligible_miners_for_scoring(self, current_challenge_all_miner_hotkeys, miner_uids, hashes, is_good_list)
    
    all_uids_to_query = [uid for uid, is_good in zip(miners_uids, miners_is_good) if is_good]
    hashes_from_uids_to_query = [hash for hash, is_good in zip(miners_hashes, miners_is_good) if is_good]

    if len(all_uids_to_query) == 0:
        bt.logging.warning(f"[run_final_prediction_phase] No miners registered after v2 without penalties found for sample {sample.variable} {sample.start_timestamp} {sample.end_timestamp}. Skipping.")
        return

    bad_uids = set([uid for uid, is_good in zip(miners_uids, miners_is_good) if not is_good])

    bad_miners_data = []
    if len(bad_uids) > 0:
        bt.logging.debug(f"{len(bad_uids)} miners did not commit or registered prior v2: {bad_uids}")
        bad_miners_data = _build_bad_miners_data(self, bad_uids)

    expected_shape = sample.output_data.shape

    def _score_batch(miner_uids, axons_to_query, compressed_predictions):
        return calculate_rmses(self, sample, miner_uids, axons_to_query, compressed_predictions, expected_shape)

    good_miners_data, predictions_bad_miners_data = await run_prediction_phase(
        self, sample, all_uids_to_query, hashes_from_uids_to_query,
        process_batch=_score_batch,
    )

    all_miners_data = good_miners_data + bad_miners_data + predictions_bad_miners_data
    complete_challenge(self, sample, all_miners_data)

    return all_miners_data

def _select_top_k_miners_to_query(best_10_hotkeys, good_hashing_hotkeys, good_hashes, lookup, sample_str):
    """Select top K miners to query, falling back to random selection if needed.
    
    Args:
        best_10_hotkeys: List of top 10 miner hotkeys to prefer
        good_hashing_hotkeys: List of all good hashing miner hotkeys
        good_hashes: List of hashes corresponding to good_hashing_hotkeys
        lookup: Dictionary mapping hotkeys to UIDs
        sample_str: String representation of the sample for logging
        
    Returns:
        Tuple of (hotkeys_to_query, hashes_of_queried, uids_to_query, query_random_miners)
        where query_random_miners indicates if random selection was used
    """
    hotkeys_to_query = []
    hashes_of_queried = []
    uids_to_query = []
    for hotkey in best_10_hotkeys:
        try:
            index = good_hashing_hotkeys.index(hotkey)
            uid = lookup[hotkey]

            uids_to_query.append(uid)
            hotkeys_to_query.append(good_hashing_hotkeys[index])
            hashes_of_queried.append(good_hashes[index])
        except Exception as e:
            bt.logging.warning(f"[_select_top_k_miners_to_query] Error: {e}")
            bt.logging.warning(f"[_select_top_k_miners_to_query] Hotkey {hotkey} not found in good hashing hotkeys. Skipping.")
            continue

    query_random_miners = False
    if hotkeys_to_query == []:
        bt.logging.warning(f"[_select_top_k_miners_to_query] No top10 hashing miners found for sample {sample_str}. Querying random miners.")
        top_k = len(best_10_hotkeys)
        len_good = len(good_hashing_hotkeys)

        if top_k > 0:
            if len_good < top_k: top_k = len_good # if there are less good hashing miners than the top k, query all good hashing miners
            indices = random.sample(range(len_good), top_k)
        else:
            if len_good > 10:
                indices = random.sample(range(len_good), 10)
            else:
                indices = range(len_good)

        hotkeys_to_query = [good_hashing_hotkeys[i] for i in indices]
        hashes_of_queried = [good_hashes[i] for i in indices]
        uids_to_query = [lookup[hotkey] for hotkey in hotkeys_to_query]
        query_random_miners = True
    
    return hotkeys_to_query, hashes_of_queried, uids_to_query, query_random_miners

def find_allowed_miners_to_query(new_hotkeys2uids, previous_hotkeys2uids):
    """Find miners that are allowed to query based on hotkey-UID consistency.
    
    Args:
        new_hotkeys2uids: Current dictionary mapping hotkeys to UIDs
        previous_hotkeys2uids: Previous dictionary mapping hotkeys to UIDs
        
    Returns:
        Tuple of (allowed_hotkeys_to_query, allowed_uids_to_query) for miners
        whose hotkey-UID mapping hasn't changed
    """
    allowed_hotkeys_to_query = []
    allowed_uids_to_query = []
    for new_hotkey, new_uid in new_hotkeys2uids.items():
        if new_hotkey in previous_hotkeys2uids and previous_hotkeys2uids[new_hotkey] == new_uid:
            allowed_hotkeys_to_query.append(new_hotkey)
            allowed_uids_to_query.append(new_uid)
        else:
            bt.logging.warning(f"Hotkey {new_hotkey} not found in previous hotkeys2uids. Skipping.")
            continue

    return allowed_hotkeys_to_query, allowed_uids_to_query

def _get_best_hotkeys_to_query(state_per_challenge: dict, state_key: str) -> Tuple[List[str], List[str]]:
    """Determine the best hotkeys to query for a given challenge, handling wind component unions.
    
    When calculating the wind speed, it is best to have the wind components come from the same miner.
    However, in a rare case, the best miner of wind u is not the best miner of wind v, and vice versa.
    Therefore, we need to take the union of the best 10 miners of the two wind components.
    """
    def get_union_hotkeys(key: str) -> Tuple[List[str], List[str]]:
        best_10 = state_per_challenge[key].best_10_miners
        var_name = key.split("@")[0]
        
        if var_name == "100m_u_component_of_wind":
            key_v = key.replace("100m_u_component_of_wind", "100m_v_component_of_wind")
            best_10_v = state_per_challenge[key_v].best_10_miners if key_v in state_per_challenge else []
            to_query = list(set(best_10 + best_10_v))
        elif var_name == "100m_v_component_of_wind":
            key_u = key.replace("100m_v_component_of_wind", "100m_u_component_of_wind")
            best_10_u = state_per_challenge[key_u].best_10_miners if key_u in state_per_challenge else []
            to_query = list(set(best_10 + best_10_u))
        else:
            to_query = best_10
            
        return best_10, to_query

    if state_key in state_per_challenge:
        best_10_hotkeys, best_hotkeys_to_query = get_union_hotkeys(state_key)
        bt.logging.info(f"The best miners for challenge {state_key} are {best_10_hotkeys}, and the ones that we query are {best_hotkeys_to_query}")
        return best_10_hotkeys, best_hotkeys_to_query
        
    # In the case when the 15 day challenge doesn't have best miner, we want to query the best miners from the 28 hour challenge. 
    matching_keys = [k for k in state_per_challenge if k.split("@")[0] == state_key.split("@")[0]]
    if matching_keys:
        matching_key = matching_keys[0]
        best_10_hotkeys, best_hotkeys_to_query = get_union_hotkeys(matching_key)
        bt.logging.info(f"The best miners for challenge {state_key} are not found, using {matching_key} instead. Best 10 miners are {best_10_hotkeys}, and the ones that we query are {best_hotkeys_to_query}")
        return best_10_hotkeys, best_hotkeys_to_query

    return [], []

def filter_good_hashing_miners_data(good_hashes, good_hotkeys, allowed_hotkeys_to_query):
    """Filter miner data to only include miners with allowed hotkeys.
    
    Args:
        good_hashes: List of good hashes
        good_hotkeys: List of good hotkeys
        allowed_hotkeys_to_query: List of allowed hotkeys to filter by
        
    Returns:
        Filtered list of good hashes, good hotkeys
    """
    filtered_hashes = []
    filtered_hotkeys = []
    for hashed_prediction, hotkey in zip(good_hashes, good_hotkeys):
        if hotkey in allowed_hotkeys_to_query:
            filtered_hashes.append(hashed_prediction)
            filtered_hotkeys.append(hotkey)

    return filtered_hashes, filtered_hotkeys

async def run_initial_prediction_top_k_phases(self, challenges, previous_hotkeys2uids):
    """Run initial prediction phase querying top K miners for each challenge.
    
    Args:
        self: BaseValidatorNeuron instance
        challenges: List of Era5Sample challenges to process
        good_hashing_miners_data_per_challenge: Dictionary mapping challenge strings to lists of MinerData
        previous_hotkeys2uids: Previous dictionary mapping hotkeys to UIDs for consistency checking
    """
   
    bt.logging.debug("[run_initial_prediction_top_k_phases] Process starting.")
    new_hotkeys2uids = {axon.hotkey: uid for uid, axon in enumerate(self.metagraph.axons)}
    allowed_hotkeys_to_query, allowed_uids_to_query = find_allowed_miners_to_query(new_hotkeys2uids, previous_hotkeys2uids)


    for sample in challenges:
        # TODO : only do if a challenge is for 15 days, we do not request the other ones. 
        if sample.predict_hours <= 49:
            bt.logging.info(f"[run_initial_prediction_top_k_phases] Skipping sample {sample.state_key} because it has {sample.predict_hours} hours, not 15 days.")
            continue
        sample_str = str(sample)

        good_hashes, good_hotkeys = self.database.get_hashing_data_for_sample(sample)
        bt.logging.info(f"[run_initial_prediction_top_k_phases] Good hashes: {good_hashes} Good hotkeys: {good_hotkeys}")
        if good_hashes == [] or good_hotkeys == []:
            bt.logging.warning(f"[run_initial_prediction_top_k_phases] No good hashing miners found for sample {sample_str}. Skipping.")
            continue

        filtered_good_hashes, filtered_good_hotkeys = filter_good_hashing_miners_data(good_hashes, good_hotkeys, allowed_hotkeys_to_query)
        
        state_key = sample.state_key
        best_10_hotkeys, best_hotkeys_to_query = _get_best_hotkeys_to_query(self.state_per_challenge, state_key)
       
        hotkeys_to_query, hashes_of_queried, uids_to_query, query_random_miners = _select_top_k_miners_to_query(best_hotkeys_to_query, filtered_good_hotkeys, filtered_good_hashes, new_hotkeys2uids, sample_str)

        def _save_and_free(miner_uids, axons_to_query, compressed_predictions):
            batch = [
                MinerData(uid=uid, hotkey=axon.hotkey, prediction=pred)
                for uid, axon, pred in zip(miner_uids, axons_to_query, compressed_predictions)
                if pred is not None
            ]
            for miner in batch:
                try:
                    save_best_miner_prediction(self, sample, miner, query_random_miners, best_10_hotkeys)
                except Exception as e:
                    bt.logging.warning(f"[run_initial_prediction_top_k_phases] {miner.hotkey} {miner.uid} Error saving prediction: {e}")
                miner.prediction = None
            return batch

        good_miners_data, bad_miners_data = await run_prediction_phase(
            self, sample, uids_to_query, hashes_of_queried,
            process_batch=_save_and_free,
        )

        bad_hotkeys = [miner.hotkey for miner in bad_miners_data]
        if len(bad_hotkeys) > 0:
            successful_insertion = self.database.mark_miners_as_bad(sample, bad_hotkeys)
            if successful_insertion:
                bad_uids = [miner.uid for miner in bad_miners_data]
                bt.logging.success(f"[run_initial_prediction_top_k_phases] Storing bad miners in SQLite database: {bad_uids}")

        self.performance_database_api.insert_top_k_info(sample, hotkeys_to_query, uids_to_query, bad_hotkeys)


async def run_prediction_phase(
    self,
    sample,
    all_uids_to_query,
    hashes_from_uids_to_query,
    process_batch: Callable[[List[int], List, List[Optional[bytes]]], List[MinerData]],
):
    """Run prediction phase querying miners in batches and verifying their predictions.

    The caller controls what happens to each batch via `process_batch`.  This
    function only owns the batch loop, UID tracking and bad-miner detection.
    
    Args:
        self: BaseValidatorNeuron instance
        sample: Era5Sample challenge to process
        all_uids_to_query: List of miner UIDs to query
        hashes_from_uids_to_query: List of expected hashes for each UID
        process_batch: Strategy callable invoked per batch with
            (miner_uids, axons_to_query, compressed_predictions).
            Must return the list of good MinerData from that batch.
        
    Returns:
        Tuple of (good_miners_data, bad_miners_data) where good_miners_data contains
        MinerData objects with valid predictions, and bad_miners_data contains
        MinerData objects for miners that failed
    """
    spec = self.challenge_registry.get(sample.state_key)
    if spec is None:
        bt.logging.error(f"[run_prediction_phase] No ChallengeSpec for state_key={sample.state_key}")
        return [], []
    settings = spec.prediction_dendrite_settings
    bt.logging.info(f'[run_prediction_phase] The number of all miners uids is {len(all_uids_to_query)}')
    steps_to_see_everyone_once = math.ceil(len(all_uids_to_query) / settings.response_batch_k)

    forward_max_uid_attempts = steps_to_see_everyone_once * settings.attempts_per_miner
    
    good_miners_uids: set[int] = set()
    good_miners_data = []
    bad_miners_uids = set()
    uid_to_hash = dict(zip(all_uids_to_query, hashes_from_uids_to_query))


    self.uid_tracker.init_count_map()
    for attempt in range(forward_max_uid_attempts):
        # Exclude only good_miners so we never re-query them; bad miners can be sampled again
        miner_uids = self.uid_tracker.get_next_batch(
            k=settings.response_batch_k,
            good_miners_uids=good_miners_uids,
            allowed_uids=all_uids_to_query,
            allowed_attempts=settings.attempts_per_miner
        )

        axons_to_query = [self.metagraph.axons[uid] for uid in miner_uids]
  

        if not miner_uids:
            bt.logging.info("[run_prediction_phase] No miners in this batch have a valid prediction. Skipping.")
            continue

        hashes_for_batch = [uid_to_hash.get(uid, None) for uid in miner_uids]

        bt.logging.debug(f"[run_prediction_phase] axons_to_query: {len(axons_to_query)} hashes :{len(hashes_from_uids_to_query)}")
        compressed_predictions = await fetch_predictions_and_verify_hashes(self, sample, hashes_for_batch, axons_to_query)

        batch_good = process_batch(miner_uids, axons_to_query, compressed_predictions)

        good_miners_data.extend(batch_good)
        good_miners_uids.update({m.uid for m in batch_good})

        if len(good_miners_uids) >= len(all_uids_to_query):
            bt.logging.success(f"All {len(all_uids_to_query)} available miners are good; stopping batch loop.")
            break

        # Introduce a delay to prevent spamming requests
        time.sleep(3)

    bad_miners_uids = set(all_uids_to_query) - good_miners_uids    
    bad_miners_data = []
    if len(bad_miners_uids) > 0:
        bt.logging.warning(f"[run_prediction_phase] {len(bad_miners_uids)} miners did not respond")
        bt.logging.debug(f"Those miners are : {bad_miners_uids}")
        bad_miners_data = _build_bad_miners_data(self, bad_miners_uids)
    
    return good_miners_data, bad_miners_data



