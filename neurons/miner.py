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
import requests
import h3

import numpy as np
from zeus.data.era5.converter import get_converter
from zeus.utils.time import to_timestamp
from zeus.validator.constants import MechanismType, WEATHERXM_CELL_RESOLUTION
from zeus.protocol import PreferenceSynapse, TimePredictionSynapse, LocalPredictionSynapse
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

        # Note that the preference synapse decides what you will be queried for,
        # so you will never have to respond to the other synapse.
        self.axon.attach(
            forward_fn=self.forward_preference,
            blacklist_fn=self.blacklist_preference
        ).attach(
            forward_fn=self.forward_era5,
            blacklist_fn=self.blacklist_era5,
            priority_fn=self.priority_era5,
        ).attach(
            forward_fn=self.forward_weatherxm,
            blacklist_fn=self.blacklist_weatherxm,
            priority_fn=self.priority_weatherxm,
        )

        self.openmeteo_api = openmeteo_requests.Client()

    async def forward_preference(self, synapse: PreferenceSynapse) -> PreferenceSynapse:
        """
        Set which mechanism you want to participate in.
        This is queried every 20 forward cycles.

        Note that validators will only ever set weights for one mechanism at a time,
        so by switching mechanism you remove all previous incentive from the previous mechanism
        """
        # TODO (miner) change this if you want to change mechanism
        synapse.mechanism = MechanismType.ERA5
        bt.logging.info(f"Setting preference to {synapse.mechanism.name}!")
        return synapse

    async def forward_era5(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
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
            f"[ERA5] Received request! Predicting {synapse.requested_hours} hours of {synapse.variable} for grid of shape {coordinates.shape}."
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
        bt.logging.info(f"[ERA5] Output shape is {output.shape}")

        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        return synapse
    

    async def forward_weatherxm(self, synapse: LocalPredictionSynapse) -> LocalPredictionSynapse:
        """
        Processes the incoming LocalPredictionSynapse for a prediction.

        Args:
            synapse (LocalPredictionSynapse): The synapse object containing the time range and coordinates

        Returns:
            LocalPredictionSynapse: The synapse object with the 'predictions' field set".
        """
        start_time = to_timestamp(synapse.start_time)
        end_time = to_timestamp(synapse.end_time)
        bt.logging.info(
            f"[WEATHERXM] Received request! Predicting {synapse.requested_hours} hours of {synapse.variable} at lat={synapse.latitude} and lon={synapse.longitude}."
        )
        ##########################################################################################################
        # TODO (miner) you likely want to improve over this baseline of calling OpenMeteo by changing this section
        h3cell = h3.latlng_to_cell(synapse.latitude, synapse.longitude, res=WEATHERXM_CELL_RESOLUTION)
        cell_pred = requests.get(
            f"https://pro.weatherxm.com/api/v1/cells/{h3cell}/forecast/wxmv1",
            params={
                "include": "hourly",
                "from": start_time.strftime("%Y-%m-%d"),
                "to": end_time.strftime("%Y-%m-%d"),
            },
            headers={
                "X-API-KEY": self.config.weatherxm.api_key
            }
        ).json()
        output = np.concat(
            [
                np.asarray(
                    [
                        hour_data[synapse.variable]
                        for hour_data in day['hourly']
                    ]
                )
                for day in cell_pred
            ],
        )[start_time.hour : -(23 - end_time.hour)] # slice to requested hours only
        ##########################################################################################################

        bt.logging.info(f"[WEATHERXM] Output shape is {output.shape}")
        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        return synapse
    

    async def blacklist_era5(self, synapse: TimePredictionSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)
    
    async def priority_era5(self, synapse: TimePredictionSynapse) -> float:
        return await self._priority(synapse)
    
    async def blacklist_weatherxm(self, synapse: LocalPredictionSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)
    
    async def priority_weatherxm(self, synapse: LocalPredictionSynapse) -> float:
        return await self._priority(synapse)

    async def blacklist_preference(self, synapse: PreferenceSynapse) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)
    

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(60)
