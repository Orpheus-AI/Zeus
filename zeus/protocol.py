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

from abc import ABC
from pydantic import Field
from typing import List, Tuple, Optional

import bittensor as bt
import torch

from zeus.validator.constants import MechanismType
from zeus import __version__ as zeus_version


class PreferenceSynapse(bt.Synapse):
    """
    A protocol representation which allows miners to define their preferred challenge,
    this is send to all miners at fixed intervals so they can change preference whenever desired.

    Note that you can only participate in one mechanic at a time, enforced by the validator.
    """
    mechanism: Optional[MechanismType] = Field(
        title="Preferred mechanism",
        description="Miners can only participate in one mechanism at a time. Please set it here.",
        default=None,
        frozen=False
    )

    def deserialize(self) -> MechanismType:
        return self.mechanism


class PredictionSynapse(bt.Synapse, ABC):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.
    """

    version: str = Field(
        title="Validator/Miner codebase version",
        description="Version matches the version-string of the SENDER, either validator or miner",
        default=zeus_version,
        frozen=False,
    )

    requested_hours: int = Field(
        title="Number of hours",
        description="The number of desired output hours for the prediction.",
        default=1,
        frozen=True,
    )

    start_time: float = Field(
        title="start timestamp",
        description="Starting timestamp in GMT+0 as a float",
        default=0.0,
        frozen=True,
    )

    end_time: float = Field(
        title="end timestamp",
        description="Ending timestamp in GMT+0 as a float",
        default=0.0,
        frozen=True,
    )

    def deserialize(self) -> torch.Tensor:
        """
        Deserialize the output. This method retrieves the response from
        the miner, deserializes it and returns it as the output of the dendrite.query() call.

        Returns:
        - torch.tensor: The deserialized response
        """
        return torch.tensor(self.predictions)


class TimePredictionSynapse(PredictionSynapse):
    """
    Used for recent/future prediction. Class name is frozen to maintain cross version compatibility
    """

    # Required request input, filled by sending dendrite caller.
    locations: List[List[Tuple[float, float]]] = Field(
        title="Locations to predict",
        description="Locations to predict. Represents a grid of (latitude, longitude) pairs.",
        default=[],
        frozen=True,
    )

    # Optional request output, filled by receiving axon.
    predictions: List[List[List[float]]] = Field(
        title="Prediction",
        description="The output tensor to be scored.",
        default=[],
        frozen=False,
    )

    # See https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Parameterlistings
    variable: str = Field(
        title="ERA5 variable you are asked to predict",
        description="Each request concerns a single CDS variable in long underscored form",
        default="2m_temperature",
        frozen=True,
    )


class LocalPredictionSynapse(PredictionSynapse):
    """
    Used for hyperlocal WeatherXM based forecasting.
    """

    latitude: float = Field(
        title="Latitude",
        description="Latitude you are asked to predict at.",
        default=0.0,
        frozen=True,
    )

    longitude: float = Field(
        title="Longitude",
        description="Longitude you are asked to predict at.",
        default=0.0,
        frozen=True,
    )

    elevation: float = Field(
        title="elevation",
        description="Elevation at station height.",
        default=0.0,
        frozen=True,
    )

    # Optional request output, filled by receiving axon.
    predictions: List[float] = Field(
        title="Prediction",
        description="The output list to be scored.",
        default=[],
        frozen=False,
    )

    variable: str = Field(
        title="WeatherXM variable you are asked to predict",
        description="Each request concerns a single variable",
        default="temperature",
        frozen=True,
    )



