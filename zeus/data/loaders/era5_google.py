from typing import Tuple, Optional, List
import dask
import zarr
import math
import xarray as xr
import numpy as np
import pandas as pd
import torch

from zeus.data.loaders.era5_base import Era5BaseLoader
from zeus.data.sample import Era5Sample
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
    Currently this dataset is NOT USED, as it would be too easy to lookup the correct answer for miners.
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
        self._predict_range = HISTORIC_HOURS_PREDICT_RANGE

        super().__init__(step_size=1, **kwargs)

    def load_dataset(self) -> xr.Dataset:
        dataset = xr.open_zarr(
            self.gcloud_url, chunks=None
        )  # don't chunk yet, that takes a lot of time.
        dataset = dataset[list(self.data_vars)]  # slice out anything we won't use.
        return dataset

 


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
