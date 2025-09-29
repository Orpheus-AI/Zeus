from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass
class MinerData:
    hotkey: str
    response_time: float
    prediction: torch.Tensor
    uid: Optional[int] = None  # all below are not set initially
    score: Optional[float] = None
    quality_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    rmse: Optional[float] = None
    baseline_improvement: Optional[float] = None # percentage (0-1)
    _shape_penalty: Optional[bool] = None

    @property
    def metrics(self):
        return {
             "RMSE": self.rmse,
             "score": self.score,
             "quality_score": self.quality_score,
             "efficiency_score": self.efficiency_score,
             "shape_penalty": self.shape_penalty,
             "response_time": self.response_time
         }

    @property
    def shape_penalty(self):
        return self._shape_penalty
    
    @shape_penalty.setter
    def shape_penalty(self, value: bool):
        self._shape_penalty = value
        if value:
            self.rmse = -1.0
            self.score = 0