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

from typing import Tuple, Optional, List, Union, Dict
from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
import torch
import pandas as pd

from zeus.data.base.loader import BaseDataLoader
from zeus.data.era5.sample import Era5Sample
from zeus.data.era5.converter import get_converter
from zeus.validator.constants import MechanismType
from zeus.validator.constants import (
    ERA5_DATA_VARS,
    ERA5_LATITUDE_RANGE,
    ERA5_LONGITUDE_RANGE,
    ERA5_AREA_SAMPLE_RANGE,
)


class Era5BaseLoader(BaseDataLoader):

    # Set parent class attributes
    mechanism = MechanismType.ERA5
    sample_cls = Era5Sample

    def __init__(
        self,
        lat_range: Tuple[float, float] = ERA5_LATITUDE_RANGE,
        lon_range: Tuple[float, float] = ERA5_LONGITUDE_RANGE,
        area_sample_range: Tuple[ float, float] = ERA5_AREA_SAMPLE_RANGE,
        **kwargs,

    ) -> None:
        super().__init__(data_vars=ERA5_DATA_VARS,**kwargs)
        self.lat_range = sorted(lat_range)
        self.lon_range = sorted(lon_range)

        self.area_sample_range = sorted(area_sample_range)

        self.dataset = self.preprocess_dataset(self.load_dataset())

    @abstractmethod
    def load_dataset(self, **kwargs) -> xr.Dataset:
        """
        Load the dataset from the source.
        Returns:
            xr.Dataset: The dataset.
        """
        pass

    def preprocess_dataset(self, dataset: Optional[xr.Dataset]) -> Optional[xr.Dataset]:
        """
        Preprocess the dataset to ensure the coordinates are (-90, 90) and (-180, 180) for latitude and longitude respectively.
        Returns:
            xr.Dataset: The preprocessed dataset or None if the dataset is not loaded.
        """
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
        variables: Optional[Union[str, List[str]]] = None
    ) -> torch.Tensor:
        """
        Get a sample from the dataset for a specific location and time range.
        You can specify the variable (needs to be in self.data_vars), if not all are returned

        Returns:
         - sample (torch.Tensor): The sample containing the input and output data as a 4D tensor.
        """
        variables = variables or self.data_vars
        if isinstance(variables, str):
            variables = [variables]
        shortcodes = [get_converter(variable).short_code for variable in variables]

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

        y_grid = torch.stack(
            [
                # only use to_numpy - which computes data - for relevant variable(s)
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
    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Sample a time range for a sample.
        Returns:
            pd.Timestamp: The start timestamp.
            pd.Timestamp: The end timestamp.
            int: The number of predict hours.
        """
        pass

    @abstractmethod
    def get_sample(self) -> Era5Sample:
        """
        Get a sample from the dataset.
        Returns:
            Era5Sample: The sample.
        """
        pass
