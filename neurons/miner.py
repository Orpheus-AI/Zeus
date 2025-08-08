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

import time
import torch
import typing
import bittensor as bt

import openmeteo_requests

import numpy as np
from zeus.data.converter import get_converter
from zeus.utils.config import get_device_str
from zeus.utils.time import to_timestamp
from zeus.protocol import TimePredictionSynapse
from zeus.base.miner import BaseMinerNeuron
from zeus import __version__ as zeus_version


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior.
    In particular, you should replace the forward function with your own logic.

    Currently the base miner does a request to OpenMeteo (https://open-meteo.com/) for predictions.
    You are encouraged to attempt to improve over this by changing the forward function.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        bt.logging.info("Attaching forward functions to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        
        # TODO(miner): Anything specific to your use case you can do here
        self.device: torch.device = torch.device(get_device_str())
        self.openmeteo_api = openmeteo_requests.Client()

    async def forward(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """
        Processes the incoming TimePredictionSynapse for a prediction.

        Args:
            synapse (TimePredictionSynapse): The synapse object containing the time range and coordinates

        Returns:
            TimePredictionSynapse: The synapse object with the 'predictions' field set".
        """
        # shape (lat, lon, 2) so a grid of locations
        coordinates = torch.Tensor(synapse.locations)
        start_time = to_timestamp(synapse.start_time)
        end_time = to_timestamp(synapse.end_time)
        bt.logging.info(
            f"Received request! Predicting {synapse.requested_hours} hours of {synapse.variable} for grid of shape {coordinates.shape}."
        )

        ##########################################################################################################
        # TODO (miner) you likely want to improve over this baseline of calling OpenMeteo by changing this section
        latitudes, longitudes = coordinates.view(-1, 2).T
        converter = get_converter(synapse.variable)
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": converter.om_name,
            "start_hour": start_time.isoformat(timespec="minutes"),
            "end_hour": end_time.isoformat(timespec="minutes"),
        }
        responses = self.openmeteo_api.weather_api(
            "https://api.open-meteo.com/v1/forecast", params=params, method="POST"
        )

        # get output as grid of [time, lat, lon, variables]
        output = torch.Tensor(np.stack(
            [
                np.stack(
                    [
                        r.Hourly().Variables(i).ValuesAsNumpy() 
                        for i in range(r.Hourly().VariablesLength())
                    ],
                    axis=-1
                )
                for r in responses
            ],
            axis=1
        )).reshape(synapse.requested_hours, *coordinates.shape[:2], -1)
        # [time, lat, lon] in case of single variable output
        output = output.squeeze(dim=-1)
        # Convert variable(s) to ERA5 units, combines variables for windspeed
        output = converter.om_to_era5(output)
        ##########################################################################################################
        bt.logging.info(f"Output shape is {output.shape}")

        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        return synapse
    

    async def blacklist(self, synapse: TimePredictionSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)
    
    async def priority(self, synapse: TimePredictionSynapse) -> float:
        return await self._priority(synapse)
    
    

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(30)
