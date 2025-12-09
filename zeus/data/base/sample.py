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

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import time

import torch
from zeus.protocol import PredictionSynapse
from zeus.validator.constants import MechanismType


class BaseSample(ABC):

    def __init__(
        self,
        start_timestamp: float,
        end_timestamp: float,
        variable: str,
        inserted_at: Optional[int] = None,
        output_data: Optional[torch.Tensor] = None,
        hours_to_predict: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ):
        """
        Create a datasample, either containing actual data or representing a database entry.
        """
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp


        self.variable = variable
        self.inserted_at = inserted_at or round(time.time())

        self.output_data = output_data
        self.hours_to_predict = hours_to_predict

        if output_data is not None:
            self.hours_to_predict = output_data.shape[0]
        elif hours_to_predict is None:
            raise ValueError("Either output data or predict hours must be provided.")
        
        # Usually set through separate loader
        self.baseline = baseline
        
    @abstractmethod
    def get_synapse(self) -> PredictionSynapse:
        """
        Converts the sample to a synapse which miners can predict on.
        Note that the output data is NOT set in this synapse.
        """
        pass

    @abstractmethod
    def get_bbox(self) -> Tuple[float, float, float, float]:
        """
        Returns the bounding box of the sample as:
        (lat_start, lat_end, lon_start, lon_end)
        """
        pass

    @property
    @abstractmethod
    def mechanism(self) -> MechanismType:
        """
        Returns the mechanism type of the sample.
        """
        pass