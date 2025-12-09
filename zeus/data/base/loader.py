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
from typing import Optional, Dict, Tuple, Type

import numpy as np
import torch
import pandas as pd

from zeus.data.base.sample import BaseSample
from zeus.validator.constants import MechanismType


class BaseDataLoader(ABC):

    def __init__(
            self,
            data_vars: Dict[str, float],
            predict_sample_range: Tuple[float, float],
            start_offset_range: Tuple[float, float],
    ):
        self.data_vars, data_var_probs = zip(*sorted(data_vars.items()))
        self.data_var_probs = np.array(data_var_probs)

        self.predict_sample_range = tuple(sorted(predict_sample_range))
        self.start_offset_range = tuple(sorted(start_offset_range))


    def is_ready(self) -> bool:
        """
        Returns whether the data loader is ready to sample data.
        """
        return True
    
    def sample_variable(self) -> str:
        """
        Sample a variable from the data loader using the probabilities of the variables.
        Returns:
            str: The sampled variable.
        """
        norm_probs = self.data_var_probs / self.data_var_probs.sum()
        return np.random.choice(self.data_vars, p=norm_probs)
    
    @property
    @abstractmethod
    def mechanism(self) -> MechanismType:
        """
        Returns the mechanism of the data loader.
        """
        ...
        
    @property
    @abstractmethod
    def sample_cls(self) -> Type[BaseSample]:
        """
        Returns the class of the corresponding sample.
        """
        ...

    @abstractmethod
    def get_sample(self) -> BaseSample:
        """
        Get a current sample from the dataset.

        Returns:
        - sample (BaseSample): The sample containing the bounding box and dates. Output data is not yet known.
        """
        ...

    @abstractmethod
    def get_last_available(self) -> pd.Timestamp:
        """
        Returns the last available timestamp in the dataset.
        This is used to determine if a challenge can be scored.
        Returns:
            pd.Timestamp: The last available timestamp.
        """
        ...


    @abstractmethod
    def get_output(self, sample: BaseSample) -> Optional[torch.Tensor]:
        """
        Get loaded output data for a particular sample,
         or None if this is not (yet) available.

        Returns:
        - tensor (torch.Tensor): The output data
        """
        ...


    def get_relative_age(self, sample: BaseSample) -> float:
        """
        Returns whether a sample involves a past prediction (<-1, 0>),
        or future prediction (<0, 1>), and by how much,
        normalised to the bounds of possible start and end times
        """
        if sample.end_timestamp < sample.inserted_at:
            # past 5 days prediction, note negative in offset_range so flip substraction
            age = pd.Timedelta(seconds=sample.inserted_at - sample.start_timestamp)
            relative_age = age / pd.Timedelta(hours=self.start_offset_range[0])
        else:
            age = pd.Timedelta(seconds=sample.end_timestamp - sample.inserted_at)
            relative_age = age / pd.Timedelta(hours=self.start_offset_range[1] + self.predict_sample_range[1])
        return relative_age