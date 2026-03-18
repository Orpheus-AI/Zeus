import math
import os
import random
import time
from typing import List, Optional, Tuple

import bittensor as bt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from zeus.base.validator import BaseValidatorNeuron
from zeus.data.sample import Era5Sample
from zeus.protocol import TimePredictionSynapse
from zeus.utils.compression import decompress_prediction
from zeus.utils.time import to_timestamp
from zeus.validator.constants import PREDICTION_DENDRITE_SETTINGS
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
    query_timestamp: float,
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

        miners_uids.append(miner_uid)
        miners_hashes.append(hash)
        miners_is_good.append(is_good)
        miners_hotkeys.append(challenge_hotkey)

    return miners_uids, miners_hashes, miners_is_good, miners_hotkeys



async def fetch_predictions_and_verify_hashes(self, sample: Era5Sample, hashes_list: List[str], axons_to_query: List[bt.Axon]) -> List[Optional[bytes]]:
    """
    Handles a single batch of miners. Variables here are cleared from memory 
    once the function returns to run_single_hash_challenge.
    """

 
    start_time = time.time()
    responses: List[TimePredictionSynapse] = await self.dendrite_prediction(
        axons=axons_to_query,
        synapse=sample.build_synapse(TimePredictionSynapse),
        deserialize=False,
        timeout=self.config.neuron.prediction_timeout,
    )
    end_time = time.time()
    bt.logging.success(f"[fetch_predictions_and_verify_hashes] Received {len(responses)} responses in {end_time - start_time} seconds")


    compressed_predictions = create_compressed_predictions(responses)
 
    compressed_predictions = _verify_hashes(axons_to_query, compressed_predictions, hashes_list)
    return compressed_predictions


async def run_final_prediction_phase(self, sample, current_challenge_all_miner_hotkeys, miner_uids, hashes, is_good):
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
    miners_uids, miners_hashes, miners_is_good, _ = filter_eligible_miners_for_scoring(self, current_challenge_all_miner_hotkeys, miner_uids, hashes, sample.query_timestamp, is_good)
    all_uids_to_query = [uid for uid, is_good in zip(miners_uids, miners_is_good) if is_good]
    hashes_from_uids_to_query = [hash for hash, is_good in zip(miners_hashes, miners_is_good) if is_good]

    bad_uids = set([uid for uid, is_good in zip(miners_uids, miners_is_good) if not is_good])

    bad_miners_data = []
    if len(bad_uids) > 0:
        bt.logging.debug(f"{len(bad_uids)} miners did not commit: {bad_uids}")
        bad_miners_data = _build_bad_miners_data(self, bad_uids)

    expected_shape = sample.output_data.shape

    good_miners_data, predictions_bad_miners_data = await run_prediction_phase(self, sample, all_uids_to_query, hashes_from_uids_to_query, expected_shape, calculate_metrics = True)

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
        target_variables = sample.variable
        sample_str = str(sample)

        good_hashes, good_hotkeys = self.database.get_hashing_data_for_sample(sample)
        bt.logging.info(f"[run_initial_prediction_top_k_phases] Good hashes: {good_hashes} Good hotkeys: {good_hotkeys}")
        if good_hashes == [] or good_hotkeys == []:
            bt.logging.warning(f"[run_initial_prediction_top_k_phases] No good hashing miners found for sample {sample_str}. Skipping.")
            continue

        filtered_good_hashes, filtered_good_hotkeys = filter_good_hashing_miners_data(good_hashes, good_hotkeys, allowed_hotkeys_to_query)
        
        


        if target_variables in self.state_per_variable:
            best_10_hotkeys = self.state_per_variable[target_variables].best_10_miners  
        else:
            best_10_hotkeys = []

        
        hotkeys_to_query, hashes_of_queried, uids_to_query, query_random_miners = _select_top_k_miners_to_query(best_10_hotkeys, filtered_good_hotkeys, filtered_good_hashes, new_hotkeys2uids, sample_str)


        expected_shape = torch.Size((sample.predict_hours,) + tuple(sample.x_grid.shape[:2]))
        good_miners_data, bad_miners_data = await run_prediction_phase(self, sample, uids_to_query, hashes_of_queried, expected_shape, calculate_metrics = False)
        # if len(good_miners_data) > 0:
            
        #     for i in range(len(good_miners_data)):
        #         save_best_miner_prediction(self, sample, good_miners_data[i], query_random_miners)

        bad_hotkeys = [miner.hotkey for miner in bad_miners_data]
        if len(bad_hotkeys) > 0:
            successful_insertion = self.database.mark_miners_as_bad(sample, bad_hotkeys)
            if successful_insertion:
                bad_uids = [miner.uid for miner in bad_miners_data]
                bt.logging.success(f"[run_initial_prediction_top_k_phases] Storing bad miners in SQLite database: {bad_uids}")

async def run_prediction_phase(self, sample, all_uids_to_query, hashes_from_uids_to_query, expected_shape, calculate_metrics):
    """Run prediction phase querying miners in batches and verifying their predictions.
    
    Args:
        self: BaseValidatorNeuron instance
        sample: Era5Sample challenge to process
        all_uids_to_query: List of miner UIDs to query
        hashes_from_uids_to_query: List of expected hashes for each UID
        expected_shape: Expected shape of the prediction tensor
        calculate_metrics: Boolean flag to determine if metrics should be calculated
        
    Returns:
        Tuple of (good_miners_data, bad_miners_data) where good_miners_data contains
        MinerData objects with valid predictions, and bad_miners_data contains
        MinerData objects for miners that failed
    """
    settings = PREDICTION_DENDRITE_SETTINGS
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



        hashes_for_batch = [uid_to_hash[uid] for uid in miner_uids]


        bt.logging.debug(f"[run_prediction_phase] axons_to_query: {len(axons_to_query)} hashes :{len(hashes_from_uids_to_query)}")
        compressed_predictions = await fetch_predictions_and_verify_hashes(self, sample, hashes_for_batch, axons_to_query)

        if calculate_metrics:
            batch_good = calculate_rmses(self, sample, miner_uids, axons_to_query, compressed_predictions, expected_shape)
        else:
            batch_good = [
                MinerData(uid=uid, hotkey=axon.hotkey, prediction=pred)
                for uid, axon, pred in zip(miner_uids, axons_to_query, compressed_predictions)
                if pred is not None
            ]

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



