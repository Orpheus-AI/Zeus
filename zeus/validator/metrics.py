import bittensor as bt
import torch
from typing import Optional, Tuple


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

def _weighted_rmse(prediction: torch.Tensor, output_data: torch.Tensor, weights: torch.Tensor) -> float:
    diff_squared =  (prediction - output_data).pow_(2)
    return diff_squared.mul_(weights).mean().sqrt().item()

def _weighted_mae(prediction: torch.Tensor, output_data: torch.Tensor, weights: torch.Tensor) -> float:
    diff =  (prediction - output_data).abs_()
    return diff.mul_(weights).mean().item()


def custom_rmse(
        output_data: torch.Tensor, # 3d shape (hours, latitude, longitude)
        prediction: torch.Tensor, # 3d shape (hours, latitude, longitude)
        latitude_weights: torch.Tensor, # latitude
        europe_weight: torch.Tensor, # ( latitude, longitude)
        default: Optional[float] = None, 
) -> Tuple[float, float]:
    """Calculates RMSE between miner prediction and correct output taking into account the lat"""
    try:
        if latitude_weights.ndim != 3:
            latitude_weights = latitude_weights.view(1, -1, 1)

        if europe_weight.ndim != 3:
            europe_weight = europe_weight[None, ...]
        
        europe_lat_weight = europe_weight * latitude_weights
        normalized_europe_lat_weight = europe_lat_weight / europe_lat_weight.mean()

        normalized_latitude_weights = latitude_weights / latitude_weights.mean()
        # This is fast and memory efficient because it chains in-place operations
        cosine_rmse = _weighted_rmse(prediction, output_data, normalized_latitude_weights)

        europe_weighted_rmse = _weighted_rmse(prediction, output_data, normalized_europe_lat_weight)

        #bt.logging.info(f'output_data {output_data} prediction {prediction} latitude_weights {latitude_weights} europe_weight {europe_weight}')
        #bt.logging.info(f'cosine_rmse {cosine_rmse} europe_weighted_rmse {europe_weighted_rmse}')

        return cosine_rmse, europe_weighted_rmse

    except Exception as e:
        # shape error etc
        if default is None:
            raise e
        bt.logging.warning(f"Failed to calculate custom RMSE between {output_data} and {prediction}. Returning {default} instead!")
        return float('inf'), float('inf')  

def custom_mae(
        output_data: torch.Tensor, # 3d shape (hours, latitude, longitude)
        prediction: torch.Tensor, # 3d shape (hours, latitude, longitude)
        latitude_weights: torch.Tensor,
        europe_weight: torch.Tensor, # ( latitude, longitude)
        default: Optional[float] = None,
) -> Tuple[float, float]:
    """Calculates MAE between miner prediction and correct output taking into account the lat"""
    try:
        if latitude_weights.ndim != 3:
            latitude_weights = latitude_weights.view(1, -1, 1)

        if europe_weight.ndim != 3:
            europe_weight = europe_weight[None, ...]
        
        europe_lat_weight = europe_weight * latitude_weights
        normalized_europe_lat_weight = europe_lat_weight / europe_lat_weight.mean()

        normalized_latitude_weights = latitude_weights / latitude_weights.mean()
        # This is fast and memory efficient because it chains in-place operations
        cosine_mae = _weighted_mae(prediction, output_data, normalized_latitude_weights)

        europe_weighted_mae = _weighted_mae(prediction, output_data, normalized_europe_lat_weight)

        return cosine_mae, europe_weighted_mae

    except Exception as e:
        # shape error etc
        if default is None:
            raise e
        bt.logging.warning(f"Failed to calculate custom MAE between {output_data} and {prediction}. Returning {default} instead!")
        return float('inf'), float('inf') 



