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

from typing import Tuple, Optional, List
import dask
import zarr
import math
import xarray as xr
import numpy as np
import pandas as pd
import torch

from zeus.data.era5.loaders.base import Era5BaseLoader
from zeus.data.era5.sample import Era5Sample
from zeus.validator.constants import (
    GCLOUD_ERA5_URL,
    HISTORIC_DATE_RANGE,
    HISTORIC_HOURS_PREDICT_RANGE,
    HISTORIC_INPUT_HOURS,
    MIN_INTERPOLATION_DISTORTIONS,
    MAX_INTERPOLATION_DISTORTIONS,
)


class ERA5GoogleLoader(Era5BaseLoader):
    """
    A dataloader based on historical data from the ERA5 dataset stored on Google Cloud.
    NOTE: Currently this dataset is NOT USED, as it would be too easy to lookup the correct answer for miners.
    The dataloader is provided mostly for reference, and a modified version might be implemented into the subnet in the future.
    """

    def __init__(
        self,
        gcloud_url: str = GCLOUD_ERA5_URL,
        date_range: Tuple[str, str] = HISTORIC_DATE_RANGE,
        input_hours: int = HISTORIC_INPUT_HOURS,
        **kwargs,
    ) -> None:
        self.gcloud_url = gcloud_url
        self.date_range = list(map(pd.to_datetime, sorted(date_range)))
        self.input_hours = input_hours

        super().__init__(predict_sample_range=HISTORIC_HOURS_PREDICT_RANGE, **kwargs)

    def load_dataset(self) -> xr.Dataset:
        dataset = xr.open_zarr(
            self.gcloud_url, chunks=None
        )  # don't chunk yet, that takes a lot of time.
        dataset = dataset[list(self.data_vars)]  # slice out anything we won't use.
        return dataset

    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Sample a random time range from the dataset in the form of pandas datetimes and a number of hours to predict.
        The start will always be the start of day,
        but the end could be any hour on a day after start.

        Returns:
         - start_timestamp (pd.Timestamp): The start of the time range to be predicted.
         - end_timestamp (pd.Timestamp): The end of the time range to be predicted.
         - num_predict_hours (int): The number of hours to be predicted (difference between start and end, including both limits).
        """
        latest_day = (self.date_range[1] - self.date_range[0]).days - math.ceil(
            self.predict_sample_range[1] / 24
        )
        start_timestamp = self.date_range[0] + pd.Timedelta(
            days=np.random.randint(0, latest_day)
        )
        num_predict_hours = np.random.randint(*self.predict_sample_range)
        end_timestamp = start_timestamp + pd.Timedelta(hours=num_predict_hours - 1)
        return start_timestamp, end_timestamp, num_predict_hours

    def get_sample(self) -> Era5Sample:
        """
        Get a random sample from the dataset.

        Returns:
        - sample (Era5Sample): The sample containing the input and output data.
        """
        # get a random rectangular bounding box
        lat_start, lat_end, lon_start, lon_end = self.sample_bbox()
        start_time, end_time, predict_hours = self.sample_time_range()

        data4d = self.get_data(
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
            start_time=start_time - pd.Timedelta(hours=self.input_hours),
            end_time=end_time,
        ) 
        # slice off lat, lon and flatten last dimension
        data = data4d[..., 2:].squeeze(dim=-1)

        input_data = data[:-predict_hours] # input_hours amount
        input_data = interp_distort(input_data)
        output_data = data[-predict_hours:]

        return Era5Sample(
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            input_data=input_data,
            output_data=output_data,
        )


def interp_distort(matrix: torch.Tensor, num_distortions: Optional[int] = None) -> torch.Tensor:
    """
    Interpolate the input data slightly at random locations, to prevent hash-lookups
    """
    if num_distortions is None:
        num_distortions = np.random.randint(MIN_INTERPOLATION_DISTORTIONS, MAX_INTERPOLATION_DISTORTIONS)

    for _ in range(num_distortions):
        t = np.random.randint(1, matrix.shape[0] - 2)
        lat = np.random.randint(1, matrix.shape[1] - 2)
        lon = np.random.randint(1, matrix.shape[2] - 2)
        offset_t, offset_lat, offset_lon = np.random.choice(
            [-1, 1], size=3, replace=True
        )
        alpha = np.random.uniform(0.0, 0.1)

        matrix[t, lat, lon] = (1 - alpha) * matrix[t, lat, lon] \
            + alpha * matrix[t + offset_t, lat + offset_lat, lon + offset_lon]

    return matrix
