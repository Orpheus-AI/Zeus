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
from typing import List, Optional
import numpy as np
import torch
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import (
    MAX_STUPIDITY,
    REWARD_DIFFICULTY_SCALER,  
    REWARD_IMPROVEMENT_WEIGHT,
)


def help_format_miner_output(
    correct: torch.Tensor, response: torch.Tensor
) -> torch.Tensor:
    """
    Reshape or slice miner output if it is almost the correct shape.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
       Sliced/reshaped miner output.
    """
    if correct.shape == response.shape:
        return response
    
    try:
        if response.ndim - 1 == correct.ndim and response.shape[-1] == 1:
            # miner forgot to squeeze.
            response = response.squeeze(-1)
        
        return response
    except IndexError:
        # if miner's output is so wrong we cannot even index, do not try anymore
        return response


def get_shape_penalty(correct: torch.Tensor, response: torch.Tensor) -> bool:
    """
    Compute penalty for predictions that are incorrectly shaped or contains NaN/infinities.

    Args:
        correct (torch.Tensor): The correct output tensor.
        response (torch.Tensor): The response tensor from the miner.

    Returns:
        float: True if there is a shape penalty, False otherwise
    """
    penalty = False
    if response.shape != correct.shape:
        penalty = True
    elif not torch.isfinite(response).all():
        penalty = True

    return penalty

def rmse(
        output_data: torch.Tensor,
        prediction: torch.Tensor,
) -> float:
    """Calculates RMSE between miner prediction and correct output"""
    try:
        return ((prediction - output_data) ** 2).mean().sqrt().item()
    except:
        # shape error etc
        return -1.0

def set_penalties(
    output_data: torch.Tensor,
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
        miner_data.prediction = help_format_miner_output(output_data, miner_data.prediction)
        shape_penalty = get_shape_penalty(output_data, miner_data.prediction)
        # set penalty, including rmse/reward if there is a penalty
        miner_data.shape_penalty = shape_penalty
    
    return miners_data


def get_curved_scores(
        raw_scores: List[float], 
        gamma: float, 
        cap_max: bool = True
) -> List[float]:
    """
    Given a list of raw float scores (can by any range),
    normalise them to 0-1 scores,
    and apply gamma correction to curve accordingly.

    Note that maximal error is capped at MAX_STUPIDITY * minimal_error if applicable,
    to prevent abuse through distribution shifting.

    This function assumes lower is better!
    """
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if cap_max:
        max_score = min(max_score, MAX_STUPIDITY * min_score)

    result = []
    for score in raw_scores:
        if max_score == min_score:
            result.append(1.0) # edge case, avoid division by 0
        else:
            # min to prevent getting negative score
            norm_score = (max_score - min(score, max_score)) / (max_score - min_score)
            result.append(np.power(norm_score, gamma)) # apply gamma correction
    
    return result
    

def set_rewards(
    output_data: torch.Tensor,
    miners_data: List[MinerData],
    baseline_data: Optional[torch.Tensor],
    difficulty_grid: np.ndarray,
    min_sota_delta: float
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

    # old challenges have no baseline, use 0 to make it not affect scoring.
    baseline_rmse = 0
    if baseline_data is not None:
        baseline_rmse = rmse(output_data, baseline_data)
        
    avg_difficulty = difficulty_grid.mean()
    # make difficulty [-1, 1], then go between [1/scaler, scaler]
    gamma = np.power(REWARD_DIFFICULTY_SCALER, avg_difficulty * 2 - 1)

    # compute unnormalised scores
    for miner_data in miners_data:
        miner_data.rmse = rmse(output_data, miner_data.prediction)
        improvement = baseline_rmse - miner_data.rmse - min_sota_delta
        miner_data.baseline_improvement = max(0, improvement)

    quality_scores = get_curved_scores([m.rmse for m in miners_data], gamma, cap_max=True)
    # negative since curving assumes minimal is the best
    improvement_scores = get_curved_scores([-m.baseline_improvement for m in miners_data], gamma, cap_max=False)

    for miner_data, quality, improvement in zip(miners_data, quality_scores, improvement_scores):
        miner_data.reward = (1 - REWARD_IMPROVEMENT_WEIGHT) * quality + REWARD_IMPROVEMENT_WEIGHT * improvement

    return miners_data
