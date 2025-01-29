from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import xarray as xr
from requests.exceptions import HTTPError
import os
import asyncio
import math

import numpy as np
import torch
import cdsapi
import pandas as pd
import bittensor as bt

from zeus.data.era5.era5_base import Era5BaseLoader
from zeus.data.sample import Era5Sample
from zeus.utils.time import get_today, get_timestamp
from zeus.validator.constants import (
    ERA5_CACHE_DIR,
    COPERNICUS_ERA5_URL,
    ERA5_START_SAMPLE_STD,
)


class Era5CDSLoader(Era5BaseLoader):

    ERA5_DELAY_DAYS = 5
        
    def __init__(
            self,
            validator_config: bt.Config,
            cache_dir: Path = ERA5_CACHE_DIR,
            **kwargs,
    ) -> None:
        self.client = cdsapi.Client(url=COPERNICUS_ERA5_URL, key=validator_config.cds.api_key, quiet=True, progress=False)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir: Path = cache_dir
        self.last_stored_timestamp: pd.Timestamp = pd.Timestamp(0)
        self.updater_running = False

        super().__init__(**kwargs)

    def is_ready(self) -> bool:
        """
        Returns whether the cache is up to date, and we can therefore sample safely.

        If not, it will start an async updating process (if it hasn't already started).
        """
        cut_off = get_today("h") - pd.Timedelta(days=self.ERA5_DELAY_DAYS)
        if self.last_stored_timestamp >= cut_off:
            return True
        
        if not self.updater_running:
            bt.logging.info("ERA5 cache is not up to date, starting updater...")
            self.updater_running = True
            asyncio.create_task(self.update_cache())
        return False
    
    def load_dataset(self) -> Optional[xr.Dataset]:
        era5_files = self.cache_dir.glob('*.nc')
        datasets = [
                xr.open_dataset(
                    fname,
                    engine="netcdf4"
                )
                for fname in era5_files

        ]
        if not datasets:
            return None

        dataset = xr.concat(datasets, "valid_time")
        dataset = dataset.sortby("valid_time")
        self.last_stored_timestamp = pd.Timestamp(dataset.valid_time.max().values)
        return dataset
    
    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Sample random start and end times according to the provided ranges.
        """
        num_predict_hours = np.random.randint(*self.predict_sample_range)

        # see visualisation in whitepaper for an elaborate explanation: basically make it more likely to sample beginning of time_sample_range
        start_offset = np.abs(int(np.random.normal(0, ERA5_START_SAMPLE_STD))) + self.min_start_offset_hours

        start_timestamp = get_today("h") + pd.Timedelta(hours=start_offset)
        end_timestamp = start_timestamp + pd.Timedelta(hours=num_predict_hours)

        return start_timestamp, end_timestamp, num_predict_hours
    
    def get_sample(self) -> Era5Sample:
        """
        Get a current sample from the dataset.

        Returns:
        - sample (Era5Sample): The sample containing the bounding box and dates. Output data is not yet known.
        """
        lat_start, lat_end, lon_start, lon_end = self.sample_bbox()
        start_time, end_time, predict_hours = self.sample_time_range()

        return Era5Sample(
            lat_start=lat_start,
            lat_end=lat_end,
            lon_start=lon_start,
            lon_end=lon_end,
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            predict_hours=predict_hours
        )

    def get_output(self, sample: Era5Sample) -> Optional[torch.Tensor]:
        end_time = get_timestamp(sample.end_timestamp)
        if end_time > self.last_stored_timestamp:
            return None
        
        start_time = end_time - pd.Timedelta(hours=sample.predict_hours - 1)
        data4d: torch.Tensor = self.get_data(
            *sample.get_bbox(), 
            start_time=start_time, 
            end_time=end_time,
        )
        # Slice off the latitude and longitude for the output
        return data4d[..., 2:].squeeze(dim=-1)

    def get_file_name(self, timestamp: pd.Timestamp) -> str:
        return os.path.join(self.cache_dir, f"era5_{timestamp.strftime('%Y-%m-%d')}.nc")

    async def download_era5_day(self, timestamp: pd.Timestamp):
            request = {
                "product_type": ["reanalysis"],
                "variable": self.data_vars,
                "year": [str(timestamp.year)],
                "month": [str(timestamp.month).zfill(2)],
                "day": [str(timestamp.day).zfill(2)],
                "time": [
                    "00:00", "01:00", "02:00",
                    "03:00", "04:00", "05:00",
                    "06:00", "07:00", "08:00",
                    "09:00", "10:00", "11:00",
                    "12:00", "13:00", "14:00",
                    "15:00", "16:00", "17:00",
                    "18:00", "19:00", "20:00",
                    "21:00", "22:00", "23:00"
                ],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            try:
                filename = self.get_file_name(timestamp)
                self.client.retrieve("reanalysis-era5-single-levels", request, target=filename)
                bt.logging.info(f"Downloaded ERA5 data for {timestamp.strftime('%Y-%m-%d')} to {filename}")
            except Exception as e:
                # Most errors can occur and should continue, but force validators to authenticate.
                if isinstance(e, HTTPError) and e.response.status_code == 401:
                    raise ValueError(f"Failed to authenticate with Copernicus API! Please specify an API key from https://cds.climate.copernicus.eu/how-to-api")

    async def update_cache(self):
        current_day = get_today("D")
        tasks = []
        expected_files = set()

        for days_ago in range(self.ERA5_DELAY_DAYS, self.ERA5_DELAY_DAYS + math.ceil(self.predict_sample_range[1] / 24) + 1):
            timestamp = current_day - pd.Timedelta(days=days_ago)
            filename = self.get_file_name(timestamp)
            expected_files.add(filename)
            # always download the five days ago file since its hours might have been updated.
            if not os.path.isfile(filename) or days_ago == self.ERA5_DELAY_DAYS:
                tasks.append(self.download_era5_day(timestamp))

        await asyncio.gather(*tasks)
        self.dataset = self.preprocess_dataset(self.load_dataset())

        if not self.is_ready():
            bt.logging.error("ERROR: ERA5 current cache update failed! This means we cannot send live challenges to miners. If this keeps occuring, please contact us on Discord.")

        # remove any old cache.
        for file in self.cache_dir.glob('*.nc'):
            if str(file) not in expected_files:
                os.remove(file)

        self.updater_running = False