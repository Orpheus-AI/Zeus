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

from typing import Tuple
import torch

from zeus.protocol import LocalPredictionSynapse
from zeus.data.base.sample import BaseSample
from zeus.validator.constants import MechanismType
from zeus import __version__ as zeus_version

class WeatherXMSample(BaseSample):

    mechanism = MechanismType.WEATHER_XM

    def __init__(
        self,
        lat: float,
        lon: float,
        elevation: float,
        station_id: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.station_id = station_id

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """
        Returns the bounding box of the sample. 
        Note that a station only has one point, so the start and end are the same.
        This format is used for backwards compatibility with the base sample class.
        Returns:
            float: The latitude.
            float: The latitude.
            float: The longitude.
            float: The longitude.
        """
        return self.lat, self.lat, self.lon, self.lon

    def get_synapse(self) -> LocalPredictionSynapse:
        """
        Converts the sample to a synapse which miners can predict on.
        Note that the output data is NOT set in this synapse.
        """
        return LocalPredictionSynapse(
            version=zeus_version,
            latitude=self.lat,
            longitude=self.lon,
            elevation=self.elevation,
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            requested_hours=self.hours_to_predict,
            variable=self.variable
        )