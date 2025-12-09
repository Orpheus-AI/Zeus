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
    def metrics(self) -> Dict[str, Optional[float]]:
        """
        Get the metrics of the miner.
        Returns:
            Dict[str, Optional[float]]: The metrics.
        """
        return {
             "RMSE": self.rmse,
             "score": self.score,
             "quality_score": self.quality_score,
             "efficiency_score": self.efficiency_score,
             "shape_penalty": self.shape_penalty,
             "response_time": self.response_time
         }

    @property
    def shape_penalty(self) -> bool:
        """
        Get the shape penalty of the miner.
        """
        return self._shape_penalty
    
    @shape_penalty.setter
    def shape_penalty(self, value: bool):
        """
        Set the shape penalty of the miner.
        """
        self._shape_penalty = value
        if value:
            self.rmse = -1.0
            self.score = 0