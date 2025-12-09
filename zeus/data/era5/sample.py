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

from typing import Optional, Tuple
import torch

from zeus.utils.coordinates import get_grid
from zeus.protocol import TimePredictionSynapse
from zeus.data.base.sample import BaseSample
from zeus.validator.constants import MechanismType
from zeus import __version__ as zeus_version


class Era5Sample(BaseSample):

    mechanism = MechanismType.ERA5

    def __init__(
        self,
        lat_start: float,
        lat_end: float,
        lon_start: float,
        lon_end: float,
        ifs_hres_baseline: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lon_start = lon_start
        self.lon_end = lon_end

        self.x_grid = get_grid(lat_start, lat_end, lon_start, lon_end)
        
        # Usually set through OpenMeteoLoader
        self.ifs_hres_baseline = ifs_hres_baseline
        

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """
        Returns the bounding box of the sample.
        Returns:
            float: The latitude start.
            float: The latitude end.
            float: The longitude start.
            float: The longitude end.
        """
        return self.lat_start, self.lat_end, self.lon_start, self.lon_end

    def get_synapse(self) -> TimePredictionSynapse:
        """
        Converts the sample to a synapse which miners can predict on.
        Note that the output data is NOT set in this synapse.
        """
        return TimePredictionSynapse(
            version=zeus_version,
            locations=self.x_grid.tolist(),
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            requested_hours=self.hours_to_predict,
            variable=self.variable
        )