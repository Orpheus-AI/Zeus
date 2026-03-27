import bittensor as bt
import torch
from typing import Optional


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



def custom_rmse(
        output_data: torch.Tensor, # 3d shape (hours, latitude, longitude)
        prediction: torch.Tensor, # 3d shape (hours, latitude, longitude)
        latitude_weights: torch.Tensor,
        europe_weight: torch.Tensor = None,
        default: Optional[float] = None,
) -> float:
    """Calculates RMSE between miner prediction and correct output taking into account the lat"""
    try:
        sqrt_diff = ((prediction - output_data) ** 2)
        if latitude_weights.ndim != 3:
            latitude_weights = latitude_weights[None, :, None]
        result = sqrt_diff * latitude_weights
        
        cosine_rmse =  result.mean().sqrt().item()

        if europe_weight is not None and europe_weight.ndim != 3:
            europe_weight = europe_weight[None, ...]

        europe_rmse = result * europe_weight
        europe_weighted_rmse = europe_rmse.mean().sqrt().item()

        return cosine_rmse, europe_weighted_rmse

    except Exception as e:
        # shape error etc
        if default is None:
            raise e
        bt.logging.warning(f"Failed to calculate custom RMSE between {output_data} and {prediction}. Returning {default} instead!")
        return default 

def custom_mae(
        output_data: torch.Tensor, # 3d shape (hours, latitude, longitude)
        prediction: torch.Tensor, # 3d shape (hours, latitude, longitude))
        latitude_weights: torch.Tensor,
        europe_weight: torch.Tensor = None,
        default: Optional[float] = None,
) -> float:
    """Calculates MAE between miner prediction and correct output taking into account the lat"""
    try:
        abs_diff = abs(prediction - output_data)
        if latitude_weights.ndim != 3:
            latitude_weights = latitude_weights[None, :, None]
        result = abs_diff * latitude_weights
        
        cosine_mae = result.mean().item()

        if europe_weight is not None and europe_weight.ndim != 3:
            europe_weight = europe_weight[None, ...]

        europe_mae = result * europe_weight
        europe_weighted_mae = europe_mae.mean().item()

        return cosine_mae, europe_weighted_mae

    except Exception as e:
        # shape error etc
        if default is None:
            raise e
        bt.logging.warning(f"Failed to calculate MAE between {output_data} and {prediction}. Returning {default} instead!")
        return default 

