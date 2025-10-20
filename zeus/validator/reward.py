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
from typing import List, Optional, Union
from traceback import format_exception
from operator import mul, truediv
import numpy as np
import torch
import bittensor as bt
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import (
    REWARD_DIFFICULTY_SCALER,
    REWARD_RMSE_WEIGHT,
    REWARD_EFFICIENCY_WEIGHT,
    MIN_RELATIVE_SCORE,
    MAX_RELATIVE_SCORE,
    CAP_FACTOR_EFFICIENCY,
    EFFICIENCY_THRESHOLD
)


def help_format_miner_output(
    correct_shape: torch.Size, response: torch.Tensor
) -> torch.Tensor:
    """
    Reshape or slice miner output if it is almost the correct shape.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
       Sliced/reshaped miner output.
    """
    if correct_shape == response.shape:
        return response
    
    try:
        if response.ndim - 1 == len(correct_shape) and response.shape[-1] == 1:
            # miner forgot to squeeze.
            response = response.squeeze(-1)
        
        return response
    except IndexError:
        # if miner's output is so wrong we cannot even index, do not try anymore
        return response


def get_shape_penalty(correct_shape: torch.Size, response: torch.Tensor) -> bool:
    """
    Compute penalty for predictions that are incorrectly shaped or contains NaN/infinities.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
        float: True if there is a shape penalty, False otherwise
    """
    penalty = False
    if response.shape != correct_shape:
        penalty = True
    elif not torch.isfinite(response).all():
        penalty = True

    return penalty

def rmse(
        output_data: torch.Tensor,
        prediction: torch.Tensor,
        default: Optional[float] = None,
) -> float:
    """Calculates RMSE between miner prediction and correct output"""
    try:
        return ((prediction - output_data) ** 2).mean().sqrt().item()
    except Exception as e:
        # shape error etc
        if default is None:
            raise e
        bt.logging.warning(f"Failed to calculate RMSE between {output_data} and {prediction}. Returning {default} instead!")
        return default 

def set_penalties(
    correct_shape: torch.Size,
    miners_data: List[MinerData],
) -> List[MinerData]:
    """
    Calculates and sets penalities for miners based on correct shape and their prediction

    Args:
        output_data (torch.Tensor): ground truth, ONLY used for shape
        miners_data (List[MinerData]): List of MinerData objects containing predictions.

    Returns:
        List[MinerData]: List of MinerData objects with penalty fields
    """
    for miner_data in miners_data:
        # potentially fix inputs for miners
        miner_data.prediction = help_format_miner_output(correct_shape, miner_data.prediction)
        shape_penalty = get_shape_penalty(correct_shape, miner_data.prediction)
        # set penalty, including rmse/reward if there is a penalty
        miner_data.shape_penalty = shape_penalty
    
    return miners_data

def get_curved_scores(
        raw_scores: List[float], 
        gamma: float, 
        max_score: float,
        cap_factor: Union[float, int] = None,
        min_score: float = None,
) -> List[float]:
    """
    Given a list of raw float scores (can by any range),
    normalise them to 0-1 scores,
    and apply gamma correction to curve accordingly.

    Note that minimal and maximal error can each capped.
     - through the cap_factor (between 0-1), so that:
       median_score * (1 - cap_factor) <= score <= median_score * (1 + cap_factor),
    - Through specifying the min and max artificially
    If neither is specified, will be the actual min/max of the scores,
        which might allow abuse through distribution shifting.

    # NOTE: This function assumes higher is better! 
    # So if you require the opposite, simply make your raw_scores their negative
    """

    if min_score is None:
        if cap_factor is None:
            # if all better than max, everyone gets perfect score here.
            min_score = min(max_score, min(raw_scores))
        
        operator = truediv if max_score > 0 else mul
        median_bound = operator(np.median(raw_scores), cap_factor)
        max_bound = operator(max_score, cap_factor)
        min_score = min(median_bound, max_bound)

    result = []
    for score in raw_scores:
        if max_score == min_score:
            result.append(1.0) # edge case, avoid division by 0
        else:
            capped_score = max(min_score, min(score, max_score))
            norm_score = (capped_score - min_score) / (max_score - min_score)
            result.append(np.power(norm_score, gamma)) # apply gamma correction
    
    return result
    

def set_rewards(
    output_data: torch.Tensor,
    miners_data: List[MinerData],
    baseline_data: Optional[torch.Tensor],
    difficulty_grid: np.ndarray,
    challenge_age: float,
    epsilon: float = 1e-12,
) -> List[MinerData]:
    """
    Calculates rewards for miner predictions based on RMSE and relative difficulty.
    NOTE: it is assumed penalties have already been scored and filtered out, 
      if not will remove them without scoring

    Args:
        output_data (torch.Tensor): The ground truth data.
        miners_data (List[MinerData]): List of MinerData objects containing predictions.
        baseline_data (torch.Tensor): OpenMeteo prediction, where additional incentive is awarded to beat this!
        difficulty_grid (np.ndarray): Difficulty grid for each coordinate.

    Returns:
        List[MinerData]: List of MinerData objects with updated rewards and metrics.
    """
    miners_data = [m for m in miners_data if not m.shape_penalty]

    if len(miners_data) == 0:
        return miners_data

    baseline_rmse = rmse(output_data, baseline_data, default=0)
    avg_difficulty = difficulty_grid.mean()
    # make difficulty [-1, 1], then go between [1/scaler, scaler]
    diff_gamma = np.power(REWARD_DIFFICULTY_SCALER, avg_difficulty * 2 - 1)
    # challenges far in future are now considered more difficult
    gamma = diff_gamma + max(0, challenge_age)

    # compute unnormalised scores
    for miner_data in miners_data:
        miner_data.rmse = rmse(output_data, miner_data.prediction)
        # zero devision proof relative improvement
        improvement = (baseline_rmse - miner_data.rmse + epsilon) / (baseline_rmse + epsilon)
        miner_data.baseline_improvement = improvement

    #  Cap between 100% worse and 80% better than OpenMeteo
    quality_scores = get_curved_scores([m.baseline_improvement for m in miners_data], gamma, min_score=MIN_RELATIVE_SCORE, max_score=MAX_RELATIVE_SCORE)
    # negative since curving assumes maximal is the best. gamma=1 since challenge content doesn't matter here.
    efficiency_scores = get_curved_scores([-m.response_time for m in miners_data], gamma=1, max_score=-EFFICIENCY_THRESHOLD, cap_factor=CAP_FACTOR_EFFICIENCY)

    for miner_data, quality, efficiency in zip(miners_data, quality_scores, efficiency_scores):
        miner_data.quality_score = quality
        miner_data.efficiency_score = efficiency
        miner_data.score = REWARD_RMSE_WEIGHT * quality + REWARD_EFFICIENCY_WEIGHT * efficiency

    return miners_data
