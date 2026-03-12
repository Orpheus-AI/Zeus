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
import math
from typing import Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np
import torch
import time
from zeus.utils.misc import split_list
from zeus.data.sample import Era5Sample
from zeus.utils.compression import decompress_prediction
from zeus.validator.constants import PERCENTAGE_GOING_TO_WINNER
from zeus.validator.miner_data import MinerData
from zeus.validator.metrics import custom_rmse, custom_mae


def should_apply_shape_penalty(correct_shape: torch.Size, prediction: torch.Tensor) -> bool:
    try:
        if prediction is None: 
           return True
            
        if prediction.numel() != np.prod(correct_shape):
            bt.logging.warning(f"Shape penalty: {prediction.shape} != {correct_shape}")
            shape_penalty = True
        elif not torch.isfinite(prediction).all():
            bt.logging.warning(f"Shape penalty: {prediction.shape} != {correct_shape}")
            shape_penalty = True
        else:
            shape_penalty = False

    except Exception as e:
        shape_penalty = True
        
    return shape_penalty

def calculate_competition_ranks(values: list[float], precision: int = 10) -> list[int]:
    """
    Pure logic: Transforms a sorted list of values into competition ranks.
    Input values MUST be sorted.
    """
    if not values:
        return []

    ranks = []
    current_rank = 1
    
    for i, val in enumerate(values):
        # Comparison with rounding to handle float noise
        if val == float('inf') or val is None:
            current_rank = len(values)
            ranks.append(current_rank)
            continue
        if i > 0 and round(val, precision) != round(values[i-1], precision):
            current_rank = current_rank + 1
        ranks.append(current_rank)
        
    return ranks


def set_errors(sample: Era5Sample, miner_uids: List[int], axons_to_query: List, compressed_predictions: List[bytes], expected_shape: torch.Size) -> List[MinerData]:

    output_data = sample.output_data
    output_data = output_data.to(torch.float16)
    latitude_weights = np.load("zeus/data/weights/latitude_weights_for_rmse.npy")
    latitude_weights = torch.from_numpy(latitude_weights).to(output_data.device).to(output_data.dtype)
    
    miners_data = []
    for uid, axon, prediction in zip(miner_uids, axons_to_query, compressed_predictions):
        hotkey = axon.hotkey
        if prediction is None:
            temp_tensor = None
        else:
            temp_tensor = decompress_prediction(prediction, expected_shape)
            

        is_penalized = should_apply_shape_penalty(expected_shape, temp_tensor)
        if is_penalized:
            rmse = float('inf')
            mae = float('inf')
        else:
            rmse = custom_rmse(output_data, temp_tensor, latitude_weights)
            mae = custom_mae(output_data, temp_tensor, latitude_weights) 

        if math.isnan(rmse): rmse = float('inf')
        if math.isnan(mae): mae = float('inf')

        # prediction is not needed anymore, so we set it to None to save memory
        miner_data = MinerData(uid=uid, hotkey=hotkey, prediction=None, rmse=rmse, mae = mae, shape_penalty=is_penalized)
        miners_data.append(miner_data)
    
    return miners_data

def calculate_scores(miners_data: List[MinerData]) -> List[MinerData]:
    for miner in miners_data:
        if miner.rmse is None or miner.mae is None:
            score = float('inf')
        else:	
            score = (miner.rmse + miner.mae)/2
        
        miner.score = score
    return miners_data

# the score is average of the two errors (rmse and mae), breaks ties based on rmse, mae and uid in this priority
# None metrics sort last (treated as worst)
def _sort_key(m):
    return (
        m.score if m.score is not None else float("inf"), # lower is better
        m.rmse if m.rmse is not None else float("inf"),
        m.mae if m.mae is not None else float("inf"),
        m.uid or 0,
    )

def set_rewards(
    miners_data: List[MinerData],
) -> List[MinerData]:
    """
    Calculates rewards for miner predictions based on RMSE and relative difficulty.
    NOTE: it is assumed penalties have already been scored and filtered out, 
      if not will remove them without scoring

    Args:
        output_data (torch.Tensor): The ground truth data.
        miners_data (List[MinerData]): List of MinerData objects containing predictions.
        difficulty_grid (np.ndarray): Difficulty grid for each coordinate.

    Returns:
        List[MinerData]: List of MinerData objects with updated rewards and metrics.
    """
    # 1. Calculate the score
    miners_data = calculate_scores(miners_data)
    
    sorted_miners = sorted(miners_data, key=_sort_key)
    
    # 2. Extract composite scores for the pure logic function
    scores = [m.score for m in sorted_miners]
    
    # 3. Get ranks and assign them
    ranks = calculate_competition_ranks(scores)
    
    for miner, rank in zip(sorted_miners, ranks):
        miner.score = float(rank)

    return sorted_miners

def compute_min_rank_weights(
    metagraph_size: int,
    hotkeys: List[str],
    uids: List[int],
    rank_history: Dict[str, List[float]],
    window_size: int,
    miners_hotkeys: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes weights by 
    1) calculating a rank for each hotkey by taking the avergae of their last window_size ranks from rank_history
     - if a hotkey doesn't have a rank history we give it a rank infinity.
    2) giving PERCENTAGE_GOING_TO_WINNER of the weight to the first one and logarithmically to the rest. 
    3) break ties based on the latest ranks. 

    Return
    -------
    np.ndarray
        Weight vector of shape (metagraph_size,) with PERCENTAGE_GOING_TO_WINNER going to the "best" miner.

    """
    # Step 1: Calculate average rank for each hotkey
    miners_metadata = []
    for uid, hotkey in zip(uids, hotkeys):
        if hotkey not in miners_hotkeys: continue
        history = rank_history.get(str(hotkey), [])
        
        if len(history) >= window_size:
            last_n = history[-window_size:]
            avg_rank = np.mean(last_n)
            # For tie-breaking: use ranks in reverse chronological order (most recent first)
            # This allows breaking ties by comparing second-to-last, third-to-last, etc.
            tie_breaker = tuple(reversed(last_n))
        else:
            avg_rank = float('inf')
            tie_breaker = (float('inf'),) * window_size
        
        miners_metadata.append({
            'uid': uid,
            'hotkey': hotkey,
            'avg_rank': avg_rank,
            'tie_breaker': tie_breaker
        })
    
    # Step 2: Sort by average rank, breaking ties with ranks in reverse chronological order
    # Lower rank is better, so we sort by (avg_rank, tie_breaker)
    # tie_breaker is a tuple of ranks from most recent to oldest, so tuple comparison works correctly
    miners_metadata.sort(key=lambda x: (x['avg_rank'], x['tie_breaker']))
    
    best_10_miners_hotkeys = [m["hotkey"] for m in miners_metadata[:10]]
    # Step 3: Assign weights
    weights = np.zeros(metagraph_size)
        
    # Best miner gets PERCENTAGE_GOING_TO_WINNER
    best_uid = miners_metadata[0]['uid']
    bt.logging.debug(f"The best miner has uid : {best_uid}") 
    bt.logging.debug(f"full information for this miner : {miners_metadata[0]}")
    weights[best_uid] = PERCENTAGE_GOING_TO_WINNER
    
    # Remaining weight distributed logarithmically among the rest
    remaining_weight = 1.0 - PERCENTAGE_GOING_TO_WINNER
    remaining_miners = miners_metadata[1:]
    
    if len(remaining_miners) > 0:
        # Create logarithmic weights for remaining miners (decreasing: better miners get more)
        # Using log(n+1-i) where i is the index (0-indexed from the second miner)
        # This gives decreasing weights: log(n), log(n-1), ..., log(2)
        n = len(remaining_miners)
        log_weights = []
        for i in range(n):
            log_weights.append(math.log(n + 1 - i))
        
        # Normalize logarithmic weights to sum to remaining_weight
        total_log = sum(log_weights)
        if total_log > 0:
            for i, miner_data in enumerate(remaining_miners):
                uid = miner_data['uid']
                weights[uid] = remaining_weight * (log_weights[i] / total_log)
        else:
            # Fallback: equal distribution if log weights sum to 0
            equal_weight = remaining_weight / len(remaining_miners)
            for miner_data in remaining_miners:
                uid = miner_data['uid']
                weights[uid] = equal_weight
    
    return weights, best_10_miners_hotkeys


def complete_challenge(
    self,
    sample: Era5Sample,
    miners_data: List[MinerData],
) -> None:
    """
    Complete a challenge by reward all miners. Based on hotkeys to also work for delayed rewarding.
    Note that non-responding miners (which get a penalty) have already been excluded.
    """
    correct_shape = sample.output_data.shape
    bt.logging.warning(f"complete_challenge: correct_shape: {correct_shape} miners_data: {len(miners_data)} {sample.variable}")
    miners_data = set_rewards(
        miners_data=miners_data, 
    )
    bt.logging.warning(f"miners_data: {len(miners_data)}")

    self.update_scores(
        [miner.score for miner in miners_data],
        [miner.hotkey for miner in miners_data],
        sample.variable
    )
    
    bt.logging.success(f"Scored stored challenges for uids: {[miner.uid for miner in miners_data]}")
    for miner in miners_data:
        #if miner.prediction is None: continue
        bt.logging.warning(
            f"UID: {miner.uid} |  Reward: {miner.score} rmse: {miner.rmse} | mae: {miner.mae} | score {miner.score} | Penalty: {miner.shape_penalty} "
        )


def calculate_rmses(self, sample, miner_uids, axons_to_query, compressed_predictions, expected_shape):
    start_time = time.time()
    miners_data = set_errors(sample, miner_uids, axons_to_query, compressed_predictions, expected_shape)
    end_time = time.time()
    bt.logging.success(f"Time taken to parse miners_data: {end_time - start_time} seconds")

    good_miners_list, bad_miners_list = split_list(
        miners_data, lambda m: not m.shape_penalty
    )

    if len(bad_miners_list) > 0:
        bad_uids = [miner.uid for miner in bad_miners_list]
        bt.logging.success(f"Punishing miners that got a penalty: {bad_uids}")

    good_uids = set()
    if len(good_miners_list) > 0:
        good_uids = set([m.uid for m in good_miners_list])
        bt.logging.success(
            f"Storing challenge and sensible miner responses in SQLite database: {good_uids}"
        )

    bt.logging.success(f"Miners data length: {len(miners_data)}")
 
    # Introduce a delay to prevent spamming requests
    time.sleep(10) #max(0, FORWARD_DELAY_SECONDS - (time.time() - start_forward)))
    return good_miners_list



