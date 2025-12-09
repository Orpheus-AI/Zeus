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

from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import xarray as xr
from requests.exceptions import HTTPError
import os
import asyncio
import math
from traceback import format_exception

import numpy as np
import torch
import cdsapi
import pandas as pd
import bittensor as bt

from zeus.data.era5.loaders.base import Era5BaseLoader
from zeus.data.era5.sample import Era5Sample
from zeus.utils.time import get_today, to_timestamp
from zeus.validator.constants import (
    ERA5_CACHE_DIR,
    COPERNICUS_ERA5_URL,
    HOURS_PREDICT_RANGE,
    ERA5_START_SAMPLE_STD,
    ERA5_UNIFORM_START_OFFSET_PROB,
    ERA5_START_OFFSET_RANGE
)

class Era5CDSLoader(Era5BaseLoader):

    ERA5_DELAY_DAYS = 5

    def __init__(
        self,
        cache_dir: Path = ERA5_CACHE_DIR,
        copernicus_url: str = COPERNICUS_ERA5_URL,
        start_sample_std: float = ERA5_START_SAMPLE_STD,
        uniform_start_prob: float = ERA5_UNIFORM_START_OFFSET_PROB,
        **kwargs,
    ) -> None:
        
        self.cds_api_key = os.getenv("CDS_API_KEY")
        self.client = cdsapi.Client(
            url=copernicus_url, key=self.cds_api_key, 
            quiet=True, progress=False, warning_callback=lambda _: None,
            sleep_max=10,
        )
        # temporarily muted to remove confusing warning
        self.client.warning_callback = None

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir: Path = cache_dir
        self.last_stored_timestamp: pd.Timestamp = pd.Timestamp(0)
        self.updater_running = False

        self.start_sample_std = start_sample_std
        self.uniform_start_prob = uniform_start_prob

        super().__init__(
            predict_sample_range=HOURS_PREDICT_RANGE, 
            start_offset_range=ERA5_START_OFFSET_RANGE, 
            **kwargs
        )

    def _get_era5_cutoff(self) -> pd.Timestamp:
        """
        Get the cutoff timestamp for the ERA5 data, which is 5 days ago on the hour.
        Returns:
            pd.Timestamp: The cutoff timestamp.
        """
        return get_today("h") - pd.Timedelta(days=self.ERA5_DELAY_DAYS)

    def is_ready(self) -> bool:
        """
        Returns whether the cache is up to date, and we can therefore sample safely.

        If not, it will start an async updating process (if it hasn't already started).
        """
        cut_off = self._get_era5_cutoff()
        if self.last_stored_timestamp >= cut_off and len(self.data_vars) == len(self.dataset.data_vars):
            return True

        if not self.updater_running:
            bt.logging.info("[ERA5Loader] ERA5 cache is not up to date, starting updater...")
            self.updater_running = True
            asyncio.get_event_loop() # force loop availability
            asyncio.create_task(self.update_cache())
        return False
    
    def delete_broken_files(self, files: List[Path]) -> bool:
        """
        Delete broken files from the cache, which is any that cannot be opened,
         or are missing datapoints.
        Args:
            files: List of file paths to check.
        Returns:
            bool: True if there are broken files, False otherwise.
        """
        broken_file = False
        for path in files:
            try:
                with xr.open_dataset(path, engine="h5netcdf") as data:
                    # if not last file, assure no missing hours
                    if pd.Timestamp(data.valid_time.max().values).day != self._get_era5_cutoff().day:
                        assert len(data.valid_time) == 24
            except:
                broken_file = True
                path.unlink(missing_ok=True)
        return broken_file

    def load_dataset(self) -> Optional[xr.Dataset]:
        """
        Load the dataset from the cache files.
        Returns:
            xr.Dataset: The dataset.
        """
        files = [f for f in self.cache_dir.rglob("*/*.nc")]

        if self.delete_broken_files(files=files):
            bt.logging.warning("[ERA5Loader] Found one or multiple broken .nc files! They will now be redownloaded...")
            self.last_stored_timestamp = pd.Timestamp(0) # reset so if it fails will trigger re-download
            return
        
        if not files:
            return
        
        dataset = xr.open_mfdataset(
            files, 
            combine="by_coords", 
            engine='h5netcdf',
            compat="no_conflicts",
        )

        dataset = dataset.sortby("valid_time")
        self.last_stored_timestamp = pd.Timestamp(dataset.valid_time.max().values)            
        return dataset

    def sample_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Sample random start and end times according to the provided ranges.
        """
        num_predict_hours = np.random.randint(*self.predict_sample_range)

         # see visualisation at Zeus/static/era5_start_offset_distribution.png
        if np.random.rand() > self.uniform_start_prob:
            start_offset = min(
                self.start_offset_range[1], # don't overshoot
                np.abs(
                    int(np.random.normal(0, self.start_sample_std))
                ) + self.start_offset_range[0]
            )
        else:
            start_offset = int(np.random.uniform(*self.start_offset_range))

        start_timestamp = get_today("h") + pd.Timedelta(hours=start_offset)
        end_timestamp = start_timestamp + pd.Timedelta(hours=num_predict_hours - 1)

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
            variable=self.sample_variable(),
            start_timestamp=start_time.timestamp(),
            end_timestamp=end_time.timestamp(),
            hours_to_predict=predict_hours,
        )
    
    def get_last_available(self) -> pd.Timestamp:
        """
        Get the last available timestamp for the ERA5 data.
        Returns:
            pd.Timestamp: The last available timestamp.
        """
        return self.last_stored_timestamp

    def get_output(self, sample: Era5Sample) -> Optional[torch.Tensor]:
        """
        Get the output for a sample.
        Args:
            sample: The sample to get the output for.
        Returns:
            Optional[torch.Tensor]: The output data (if available). Shape: (time, lat, lon).
        """
        end_time = to_timestamp(sample.end_timestamp)
        if end_time > self.last_stored_timestamp:
            return None

        data4d: torch.Tensor = self.get_data(
            *sample.get_bbox(),
            start_time=to_timestamp(sample.start_timestamp),
            end_time=end_time,
            variables=sample.variable
        )
        # Slice off the latitude and longitude for the output
        return data4d[..., 2:].squeeze(dim=-1)

    def get_file_name(self, variable: str, timestamp: pd.Timestamp) -> str:
        """
        Get the file name for a variable and timestamp.
        """
        return os.path.join(self.cache_dir, variable, f"era5_{timestamp.strftime('%Y-%m-%d')}.nc")

    def download_era5_day(self, variable: str, timestamp: pd.Timestamp):
        """
        Make a request to Copernicus. 
        Can only request one variable at a time for now, as it will otherwise zip them
        """
        request = {
            "product_type": ["reanalysis"],
            "variable": [variable],
            "year": [str(timestamp.year)],
            "month": [str(timestamp.month).zfill(2)],
            "day": [str(timestamp.day).zfill(2)],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        try:
            filename = self.get_file_name(variable, timestamp)
            Path(filename).parent.mkdir(exist_ok=True)
            self.client.retrieve(
                "reanalysis-era5-single-levels", request, target=filename
            )

            bt.logging.info(
                f"[ERA5Loader] Downloaded {variable} ERA5 data for {timestamp.strftime('%Y-%m-%d')} to {filename}"
            )
        except Exception as e:
            # Most errors can occur and should continue, but force validators to authenticate.
            if isinstance(e, HTTPError) and e.response.status_code == 401:
                raise ValueError(
                    f"Failed to authenticate with Copernicus API! Please specify an API key from https://cds.climate.copernicus.eu/how-to-api"
                )
            else:
                bt.logging.error(
                    f"[ERA5Loader] Failed to download {variable} ERA5 data for {timestamp.strftime('%Y-%m-%d')}: {e}"
                )

    async def update_cache(self):
        """
        Update the cache by downloading the latest ERA5 data asynchronously.
        """
        current_day = get_today("D")
        tasks = []
        expected_files = set()

        for variable in self.data_vars:
            for days_ago in range(
                self.ERA5_DELAY_DAYS,
                self.ERA5_DELAY_DAYS + math.ceil(self.predict_sample_range[1] / 24) + 1,
            ):
                timestamp = current_day - pd.Timedelta(days=days_ago)
                filename = self.get_file_name(variable, timestamp)
                expected_files.add(filename)
                # always download the five days ago file since its hours might have been updated.
                if not os.path.isfile(filename) or days_ago == self.ERA5_DELAY_DAYS:
                    tasks.append(asyncio.to_thread(self.download_era5_day, variable, timestamp))

        try:
            await asyncio.gather(*tasks)
            self.dataset = self.preprocess_dataset(self.load_dataset())
            assert self.is_ready()
            bt.logging.info("[ERA5Loader] Successfully updated cache -- ready to send challenges!")

            # remove any old cache.
            for file in self.cache_dir.rglob("*.nc"):
                if str(file) not in expected_files:
                    file.unlink(missing_ok=True)

        except Exception as err:
            bt.logging.error(f"[ERA5Loader] ERA5 cache update failed! {''.join(format_exception(type(err), err, err.__traceback__))}")
        finally:
            self.updater_running = False
