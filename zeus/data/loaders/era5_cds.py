import asyncio
import logging
import math
import os
from pathlib import Path
from traceback import format_exception
from typing import Dict, List, Optional, Tuple, Union

import bittensor as bt
import cdsapi
import pandas as pd
import torch
import xarray as xr
from requests.exceptions import HTTPError

from zeus.data.converter import get_converter
from zeus.data.loaders.era5_base import Era5BaseLoader
from zeus.data.sample import Era5Sample
from zeus.utils.time import get_today, to_timestamp
from zeus.validator.constants import (
    COPERNICUS_ERA5_URL,
    DEFAULT_STEP_SIZE,
    ERA5_CACHE_DIR,
    TIME_WINDOWS_PER_CHALLENGE,
)


class Era5CDSLoader(Era5BaseLoader):

    ERA5_DELAY_DAYS = 5

    def __init__(
        self,
        cache_dir: Path = ERA5_CACHE_DIR,
        copernicus_url: str = COPERNICUS_ERA5_URL,
        **kwargs,
    ) -> None:
        self.cds_api_key = os.getenv("CDS_API_KEY")
        self.client = cdsapi.Client(
            url=copernicus_url, key=self.cds_api_key,
            quiet=True, progress=False, warning_callback=lambda _: None,
            sleep_max=10,
        )
        self.client.warning_callback = None
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir: Path = cache_dir
        self.last_stored_timestamp: pd.Timestamp = pd.Timestamp(0)
        self.updater_running = False
        super().__init__(step_size=DEFAULT_STEP_SIZE, **kwargs)

    def _get_era5_cutoff(self) -> pd.Timestamp:
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
            bt.logging.info("ERA5 cache is not up to date, starting updater...")
            self.updater_running = True
            asyncio.get_event_loop() # force loop availability
            asyncio.create_task(self.update_cache())
        return False
    
    def delete_broken_files(self, files: List[Path]):
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
        files = [f for f in self.cache_dir.rglob("*/*.nc")]

        if self.delete_broken_files(files=files):
            bt.logging.warning("Found one or multiple broken .nc files! They will now be redownloaded...")
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

    def get_challenge_samples(self) -> List[Era5Sample]:
        """
        First challenge: hours [0,48] from base_start.
        Second challenge: hours [49, 24*15] from base_start.
        """
        samples = []

        lat_start, lat_end, lon_start, lon_end = self.get_full_bbox()
        base_start = (pd.Timestamp.now('UTC').floor('6h')).replace(tzinfo=None)
        step_size = self.step_size

        for variable in self.data_vars:
            for start_h, end_h in TIME_WINDOWS_PER_CHALLENGE:
                chunk_start = base_start + pd.Timedelta(hours=start_h)
                chunk_end = base_start + pd.Timedelta(hours=end_h)
                predict_hours = len(
                    pd.date_range(chunk_start, chunk_end, freq=f"{step_size}h")
                )
                bt.logging.info(
                    f"challenge window [{start_h}h,{end_h}h]: {chunk_start} -> {chunk_end}, "
                    f"predict_hours={predict_hours}"
                )

                samples.append(
                    Era5Sample(
                        lat_start=lat_start,
                        lat_end=lat_end,
                        lon_start=lon_start,
                        lon_end=lon_end,
                        variable=variable,
                        start_timestamp=chunk_start.timestamp(),
                        end_timestamp=chunk_end.timestamp(),
                        predict_hours=predict_hours,
                        step_size=step_size,
                        start_offset=start_h,
                        end_offset=end_h,
                    )
                )
        return samples


    def get_output(self, sample: Era5Sample) -> Optional[torch.Tensor]:
        end_time = to_timestamp(sample.end_timestamp)
        if end_time > self.last_stored_timestamp:
            return None
        data4d = self.get_data(
            *sample.get_bbox(),
            start_time=to_timestamp(sample.start_timestamp),
            end_time=end_time,
            variables=sample.variable,
            step_size=sample.step_size,
        )
        if data4d is None:
            return None
        result = data4d[..., 2:].squeeze(dim=-1)
        variable_converter = get_converter(sample.variable)
        return variable_converter.era5_to_target(result)

    def get_file_name(self, variable: str, timestamp: pd.Timestamp) -> str:
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
                f"Downloaded {variable} ERA5 data for {timestamp.strftime('%Y-%m-%d')} to {filename}"
            )
        except Exception as e:
            # Most errors can occur and should continue, but force validators to authenticate.
            if isinstance(e, HTTPError) and e.response.status_code == 401:
                raise ValueError(
                    f"Failed to authenticate with Copernicus API! Please specify an API key from https://cds.climate.copernicus.eu/how-to-api"
                )
            else:
                bt.logging.error(
                    f"Failed to download {variable} ERA5 data for {timestamp.strftime('%Y-%m-%d')}: {e}"
                )

    async def update_cache(self):
        current_day = get_today("D")
        tasks = []
        expected_files = set()

        for variable in self.data_vars:
            for days_ago in range(
                self.ERA5_DELAY_DAYS,
                self.ERA5_DELAY_DAYS +19,
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
            bt.logging.info("Successfully updated cache -- ready to send challenges!")

            # remove any old cache.
            for file in self.cache_dir.rglob("*.nc"):
                if str(file) not in expected_files:
                    file.unlink(missing_ok=True)

        except Exception as err:
            bt.logging.error(f"ERA5 cache update failed! {''.join(format_exception(type(err), err, err.__traceback__))}")
        finally:
            self.updater_running = False

  