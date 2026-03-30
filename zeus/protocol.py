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
from typing import Optional, Callable, Type, get_args

from starlette.responses import Response
import bittensor as bt
from bittensor.core.axon import AxonMiddleware
import torch
from pydantic import Field

from zeus.validator.constants import ForecastType
from zeus.utils.compression import decompress_prediction


class PredictionSynapse(bt.Synapse, ABC):
    """
    A protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.
    """

    def get_byte_field_name(self) -> Optional[str]:
        """Get name of field marked as bytes, ensures at most one can exist"""
        byte_fields = [
            name 
            for name, field in self.model_fields.items() 
            if field.annotation is bytes or bytes in get_args(field.annotation)
        ]
        assert len(byte_fields) <= 1, "Cannot dump more than one byte field!"
        return byte_fields[0] if byte_fields else None

    def get_body(self) -> Optional[bytes]:
        """Allows for getting raw bytes request body if it has a byte field"""
        return getattr(self, self.get_byte_field_name() or '', None)

    version: str = Field(
        title="Validator/Miner codebase version",
        description="Version matches the version-string of the SENDER, either validator or miner",
        default = "",
        frozen = False,
    )

    requested_hours: int = Field(
        title="Number of timestamps",
        description="The number of time steps (timestamps) in the prediction output, i.e. length of the time dimension.",
        default=1,
        frozen=True,
    )

    # See https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Parameterlistings
    variable: str = Field(
        title="ERA5 variable you are asked to predict",
        description="Each request concerns a single CDS variable in long underscored form",
        default="2m_temperature",
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

    step_size: int = Field(
        title="step size",
        description="Step size in hours",
        default=1,
        frozen=True,
    ) 

    latitude_start: float = Field(
        title="latitude start",
        description="Latitude start",
        default=-90,
        frozen=True,
    )
    latitude_end: float = Field(
        title="latitude end",
        description="Latitude end",
        default=90,
        frozen=True,
    )

    longitude_start: float = Field(
        title="longitude start",
        description="Longitude start",
        default=-180,
        frozen=True,
    )
    longitude_end: float = Field(
        title="longitude end",
        description="Longitude end",
        default=179.75,
        frozen=True,
    )

    @property
    def forecase_type(self) -> ForecastType:
        """Get forecase type (short or medium)"""
        if self.requested_hours == 49:
            return ForecastType.SHORT_TERM
        return ForecastType.MEDIUM_TERM

class TimePredictionSynapse(PredictionSynapse):
    """
    Used for recent/future prediction. Class name is frozen to maintain cross version compatibility
    """

    # Required request input, filled by sending dendrite caller.
    # locations: List[List[Tuple[float, float]]] = Field(
    #     title="Locations to predict",
    #     description="Locations to predict. Represents a grid of (latitude, longitude) pairs.",
    #     default=[],
    #     frozen=True,
    # )
    # Response output: miners must set this. Base64-encoded payload (blosc2-compressed float32).
    # Use str so JSON/transport does not try to decode binary as UTF-8.
    predictions: Optional[bytes] = Field(
        title="Prediction (blosc2)",
        description="Base64-encoded blosc2-compressed float32. deserialize(expected_shape=...) decodes to tensor (time, lat, lon).",
        default=None,
        frozen=False,
    )
    def deserialize(self, expected_shape=None) -> torch.Tensor:
        """
        Deserialize the output. Decodes base64 then blosc2 (if expected_shape given) to a tensor (time, lat, lon).

        Returns:
        - torch.tensor: The deserialized response, shape (time, lat, lon). Invalid or missing data yields an empty tensor (shape penalty).
        """
        if not self.predictions:
            return torch.tensor([])
        try:
            return decompress_prediction(self.predictions, tuple(expected_shape))
       
        except Exception:
            return torch.tensor([])

    

class HashedTimePredictionSynapse(PredictionSynapse):
    """
    Hash-commitment round: same request as PredictionSynapse.
    Response: miner sets hash = sha256(compressed_predictions + hotkey).hexdigest().
    """

    # Response field: miner sets this (frozen=False).
    hash: Optional[str] = Field(
        title="hash",
        description="sha256(compressed_predictions + miner hotkey).hexdigest()",
        default=None,
        frozen=False,
    )


def patch_synapse_response(middleware_cls: Type[AxonMiddleware]) -> Response:

    original_function = middleware_cls.synapse_to_response.__func__

    @classmethod
    async def wrapper(cls, synapse: PredictionSynapse, start_time: float, response_override: Optional[Response]=None):
        response_override = Response(
            content=synapse.get_body()
        )
        return await original_function(cls, synapse, start_time, response_override=response_override)

    
    middleware_cls.synapse_to_response = wrapper
       

        
