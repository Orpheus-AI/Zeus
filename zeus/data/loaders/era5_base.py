from typing import Tuple, Optional, List, Union, Dict
from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
import torch
import pandas as pd

from zeus.data.sample import Era5Sample
from zeus.data.converter import get_converter
from zeus.validator.constants import (
    ERA5_DATA_VARS,
    ERA5_LATITUDE_RANGE,
    ERA5_LONGITUDE_RANGE,
    ERA5_AREA_SAMPLE_RANGE,
    DEFAULT_STEP_SIZE,
)


class Era5BaseLoader(ABC):

    def __init__(
        self,
        max_time_offset: int,
        step_size: int = DEFAULT_STEP_SIZE,
        data_vars: Dict[str, float] = ERA5_DATA_VARS,
        lat_range: Tuple[float, float] = ERA5_LATITUDE_RANGE,
        lon_range: Tuple[float, float] = ERA5_LONGITUDE_RANGE,
        area_sample_range: Tuple[float, float] = ERA5_AREA_SAMPLE_RANGE,
    ) -> None:
        self.data_vars, self.data_var_probs = zip(*sorted(data_vars.items()))
        self.data_var_probs = np.array(self.data_var_probs)
        self.max_time_offset = max_time_offset
        self.step_size = step_size
        self.lat_range = sorted(lat_range)
        self.lon_range = sorted(lon_range)
        self.area_sample_range = sorted(area_sample_range)
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

    def get_full_bbox(self) -> Tuple[float, float, float, float]:
        """Return the full bounding box from lat/lon ranges (lat_start, lat_end, lon_start, lon_end)."""
        return self.lat_range[0], self.lat_range[1], self.lon_range[0], self.lon_range[1]

    def sample_variable(self) -> str:
        norm_probs = self.data_var_probs / self.data_var_probs.sum()
        return np.random.choice(self.data_vars, p=norm_probs)

    def get_data(
        self,
        lat_start: float,
        lat_end: float,
        lon_start: float,
        lon_end: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        variables: Optional[Union[str, List[str]]] = None,
        step_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get data for a location and time range. If step_size is set, time dimension
        is subsampled to every step_size-th point (e.g. 3 → 0h, 3h, 6h, ...).
        """
        variables = variables or self.data_vars
        if isinstance(variables, str):
            variables = [variables]
        shortcodes = [get_converter(variable).short_code for variable in variables]

        subset = self.dataset.sel(
            latitude=slice(lat_start, lat_end),
            longitude=slice(lon_start, lon_end),
        )
        time_dim = "valid_time" if "valid_time" in subset.dims else "time"

        
        desired_timesteps = pd.date_range(start=start_time, end=end_time, freq=f'{step_size}h')
        valid_desired_timesteps = desired_timesteps.intersection(subset[time_dim].values)

        # in the rare case that the era5 files were corrupted and we don't have all the timesteps 
        if set(desired_timesteps) != set(valid_desired_timesteps):
            return None # TODO : Check that this makes sense, I changed get_output in era5 cds as well
        
        subset = subset.sel({time_dim: valid_desired_timesteps})

        y_grid = torch.stack(
            [
                torch.as_tensor(subset[var].to_numpy(), dtype=torch.float)
                for var in shortcodes
            ],
            dim=-1,
        )  # (time, lat, lon, data_vars)

        x_grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.as_tensor(subset[v].to_numpy(), dtype=torch.float)
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
    def get_sample(self) -> Era5Sample:
        pass
