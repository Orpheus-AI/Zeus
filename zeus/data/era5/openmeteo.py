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

import os
import openmeteo_requests

import numpy as np
import torch

from zeus.data.base.predictor import BasePredictor
from zeus.data.era5.sample import Era5Sample
from zeus.data.era5.converter import get_converter
from zeus.utils.time import to_timestamp
from zeus.validator.constants import (
    OPEN_METEO_URL
)

class OpenMeteoLoader(BasePredictor):

    def __init__(
        self,
        open_meteo_url = OPEN_METEO_URL,
    ) -> None:
        
        self.api_key = os.getenv("OPEN_METEO_API_KEY")
        self.open_meteo_url = open_meteo_url
        self.open_meteo_api = openmeteo_requests.Client()

    def get_forecast(self, sample: Era5Sample, model: str = "best_match") -> torch.Tensor:
        """
        Get the forecast for a sample using the OpenMeteo API.
        Args:
            sample: The sample to get the forecast for.
            model: Optional, the model to use for the forecast. Defaults to "best_match".
        Returns:
            torch.Tensor: The forecast data. Shape: (time, lat, lon).
        """
        start_time = to_timestamp(sample.start_timestamp)
        end_time = to_timestamp(sample.end_timestamp)

        latitudes, longitudes = sample.x_grid.view(-1, 2).T
        converter = get_converter(sample.variable)
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": converter.om_name,
            "models": model,
            "start_hour": start_time.isoformat(timespec="minutes"),
            "end_hour": end_time.isoformat(timespec="minutes"),
            "apikey": self.api_key
        }

        responses = self.open_meteo_api.weather_api(
            self.open_meteo_url, params=params, method="POST"
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
        )).reshape(sample.hours_to_predict, *sample.x_grid.shape[:2], -1)
        # [time, lat, lon] in case of single variable output
        output = output.squeeze(dim=-1)
        # Convert variable(s) to ERA5 units, combines variables for windspeed
        output = converter.om_to_era5(output)
        return output
