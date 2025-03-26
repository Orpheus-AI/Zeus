from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
import torch
import pandas as pd

from zeus.data.sample import Era5Sample
from zeus.validator.constants import (
    ERA5_DATA_VARS,
    ERA5_LATITUDE_RANGE,
    ERA5_LONGITUDE_RANGE,
    ERA5_DATE_RANGE,
    ERA5_AREA_SAMPLE_RANGE,
    ERA5_START_OFFSET_RANGE,
    ERA5_HOURS_PREDICT_RANGE,
)


class Era5BaseLoader(ABC):

    def __init__(
        self,
        data_vars: List[str] = ERA5_DATA_VARS,
        lat_range: Tuple[float, float] = ERA5_LATITUDE_RANGE,
        lon_range: Tuple[float, float] = ERA5_LONGITUDE_RANGE,
        date_range: Tuple[str, str] = ERA5_DATE_RANGE,
        area_sample_range: Tuple[ float, float] = ERA5_AREA_SAMPLE_RANGE,
        start_offset_range: Tuple[int, int] = ERA5_START_OFFSET_RANGE,
        predict_sample_range: Tuple[float, float] = ERA5_HOURS_PREDICT_RANGE,
        noise_factor: float = 1e-3,
    ) -> None:
        self.data_vars = data_vars
        self.noise_factor = noise_factor

        self.lat_range = sorted(lat_range)
        self.lon_range = sorted(lon_range)
        self.date_range = list(map(pd.to_datetime, sorted(date_range)))

        self.area_sample_range = sorted(area_sample_range)
        self.start_offset_range = start_offset_range
        self.predict_sample_range = sorted(predict_sample_range)

        self.dataset = self.preprocess_dataset(self.load_dataset())

    @abstractmethod
    def load_dataset(self, **kwargs) -> xr.Dataset:
        pass

    def preprocess_dataset(self, dataset: Optional[xr.Dataset]) -> xr.Dataset:
        # ensure the coordinates are (-90, 90) and (-180, 180) for latitude and longitude respectively.
        if dataset is None:
            return None

        if dataset["longitude"].max() > 180:
            dataset = dataset.assign_coords(
                longitude=(dataset["longitude"].values + 180) % 360 - 180
            )
        if dataset["latitude"].max() > 90:
            dataset = dataset.assign_coords(latitude=dataset["latitude"].values - 90)

        dataset = dataset.sortby(["latitude", "longitude"])
        return dataset

    def sample_bbox(self, fidelity: int = 4) -> Tuple[float, float, float, float]:
        """
        Sample a bounding box that is both inside the dataset and matching the 0.25 degree grid.
        We can't assume the dataset is loaded, so have to manually convert the latitude and longitude ranges to the 0.25 degree grid.
        Area sample range is already in 0.25 degree units.

        Returns:
          - Latitude start
          - Latitude end
          - Longitude start
          - Longitude end
        """
        # make sure the lat and lon samples are fixed to a 0.25 degree grid, so 4 measurements per degree.
        lat_start = (
            np.random.randint(
                self.lat_range[0] * fidelity,
                self.lat_range[1] * fidelity - self.area_sample_range[1],
            )
            / fidelity
        )
        lat_end = lat_start + np.random.randint(*self.area_sample_range) / fidelity

        lon_start = (
            np.random.randint(
                self.lon_range[0] * fidelity,
                self.lon_range[1] * fidelity - self.area_sample_range[1],
            )
            / fidelity
        )
        lon_end = lon_start + np.random.randint(*self.area_sample_range) / fidelity
        return lat_start, lat_end, lon_start, lon_end

    def get_data(
        self,
        lat_start: float,
        lat_end: float,
        lon_start: float,
        lon_end: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> torch.Tensor:
        """
        Get a sample from the dataset for a specific location and time range.

        Returns:
         - sample (torch.Tensor): The sample containing the input and output data as a 4D tensor.
        """

        subset = self.dataset.sel(
            latitude=slice(lat_start, lat_end),
            longitude=slice(lon_start, lon_end),
        )
        # CDS NC files don't have time but 'valid_time' instead.
        if "valid_time" in subset.dims:
            # increment end time to make it inclusive
            subset = subset.sel(valid_time=slice(start_time, end_time))
        else:
            subset = subset.sel(time=slice(start_time, end_time))

        subset = subset.compute()  # heavy loading - fetch the actual data here.

        y_grid = torch.stack(
            [
                torch.as_tensor(subset[var].data, dtype=torch.float)
                for var in subset.data_vars
            ],
            dim=-1,
        )  # (time, lat, lon, data_vars)

        x_grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.as_tensor(subset[v].data, dtype=torch.float)
                    for v in ("latitude", "longitude")
                ],
                indexing="ij",
            ),
            dim=-1,
        )  # (lat, lon, 2)
        x_grid = x_grid.expand(y_grid.shape[0], *x_grid.shape)  # (time, lat, lon, 2)

        data = torch.cat([x_grid, y_grid], dim=-1)
        return data

    @abstractmethod
    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        pass

    @abstractmethod
    def get_sample(self) -> Era5Sample:
        pass
