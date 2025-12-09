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

from typing import List, Union, Dict, Any, Tuple
from functools import partial
import os
import time
import asyncio
import traceback

import bittensor as bt
import numpy as np
import pandas as pd
import torch
import uvicorn

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, Request
from timezonefinder import TimezoneFinder

from zeus.validator.constants import (
    ERA5_START_OFFSET_RANGE,
    ERA5_AREA_SAMPLE_RANGE, 
    PROXY_QUERY_K, 
    ERA5_DATA_VARS,
    WEATHER_XM_DATA_VARS,
    WEATHERXM_START_OFFSET_RANGE
)
from zeus.base.validator import BaseValidatorNeuron
from zeus.validator.reward import help_format_miner_output, get_shape_penalty
from zeus.protocol import TimePredictionSynapse, LocalPredictionSynapse
from zeus.utils.time import get_today, get_hours, safe_tz_convert
from zeus.utils.coordinates import get_grid, expand_to_grid, interp_coordinates
from zeus.data.era5.converter import get_converter, VariableConverter
from zeus.data.weatherxm.variables import get_wxm_variable, WeatherXMVariable
from zeus.validator.constants import MechanismType
from zeus.data.weatherxm.station import Station

from zeus.base.dendrite import ZeusDendrite

class ValidatorProxy:

    PROXY_TIMEOUT = 5

    def __init__(
        self,
        validator: BaseValidatorNeuron,
    ):
        self.proxy_api_key = os.getenv("PROXY_API_KEY")
        if not self.proxy_api_key or self.proxy_api_key == "":
            bt.logging.warning("[PROXY] No proxy API key has been specified in the validator.env file! " +
                                "Proxy will be disabled for safety reasons")
            return

        self.timezone_finder = TimezoneFinder()
        self.validator = validator
        self.dendrite = ZeusDendrite(wallet=validator.wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/predictGrid",
            self.predict_grid,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_api_route(
            "/predictPoint",
            self.predict_point,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_api_route(
            "/predictStation",
            self.predict_station,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )

        self.loop = asyncio.get_event_loop()
        if self.validator.config.proxy.port:
            self.start_server()

    async def get_self(self):
        return self

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app, host="0.0.0.0", port=self.validator.config.proxy.port
            )
        )
        self.executor.submit(self.server.run)

    def stop_server(self):
        if hasattr(self, "server"):
            self.server.should_exit = True
            self.executor.shutdown(wait=True)

    def authorize_token(self, headers):
        authorization: str = headers.get("authorization", None)
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        if authorization != self.proxy_api_key:
            raise HTTPException(status_code=401, detail="Invalid authorization token")

    def get_proxy_uids(self, mechanism: MechanismType) -> List[int]:
        miner_uids = np.asarray(list(self.validator.uid_tracker.get_responding_uids(mechanism)), dtype=np.int32)
        inv_ranked_uids = miner_uids[self.validator.metagraph.incentive[miner_uids].argsort()]
        sel_uids = inv_ranked_uids[-PROXY_QUERY_K:].tolist()
    
        if len(sel_uids) < PROXY_QUERY_K:
            to_sample = PROXY_QUERY_K - len(sel_uids)
            bt.logging.warning(
                    f"[PROXY] Not enough non-busy recent miners found, sampling {to_sample} additional miners"
            )
            sel_uids.extend(
                self.validator.uid_tracker.get_random_uids(
                    k = to_sample,
                    mechanism=mechanism,
                    tries = 2,
                    sleep = 0.2
                )
            )
        return sel_uids

    def safe_average(self, responses: List[torch.Tensor], correct_shape: torch.Size) -> torch.Tensor:
        responses = [r for r in responses if is_valid_response(r, correct_shape)]
        if not responses:
            bt.logging.info(f"[PROXY] Received no valid responses from miners")
            raise HTTPException(status_code=500, detail="No valid response received from miners")

        return torch.stack(responses).mean(dim=0)

    async def predict_grid(self, request: Request):
        self.authorize_token(request.headers)
        bt.logging.info("[PROXY] Received an organic request!")

        request_start = time.time()
        # catch errors to prevent log spam if API is missused
        try:
            payload = await request.json()
            lat_start = payload["lat_start"]
            lat_end = payload["lat_end"]
            
            lon_start = payload["lon_start"]
            lon_end = payload["lon_end"]

            grid = get_grid(lat_start, lat_end, lon_start, lon_end)
            assert (
                min(grid.shape[:2]) >= ERA5_AREA_SAMPLE_RANGE[0] and max(grid.shape[:2]) < ERA5_AREA_SAMPLE_RANGE[1]
            ), f"Area range invalid. With 0.25 degree fidelity, each dimension should be in {ERA5_AREA_SAMPLE_RANGE}"

            start_time, end_time, predict_hours = self._parse_time_inputs(payload, MechanismType.ERA5)
            variable_conv = self._parse_era5_variable(payload)

            synapse = TimePredictionSynapse(
                locations=grid.tolist(),
                variable=variable_conv.data_var,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                requested_hours=predict_hours,
            )

        except Exception:
            bt.logging.info(f"[PROXY] Organic request was invalid.")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request, parsing failed with error:\n {traceback.format_exc()}",
            )

        miner_uids = self.get_proxy_uids(MechanismType.ERA5)
        self.validator.uid_tracker.add_busy_uids(miner_uids, MechanismType.ERA5)
        axons = [self.validator.metagraph.axons[uid] for uid in miner_uids]

        prediction = self.safe_average(await self.dendrite(
            axons=axons,
            synapse=synapse,
            timeout=self.PROXY_TIMEOUT,
        ), correct_shape=(predict_hours, grid.shape[0], grid.shape[1]))
        self.validator.uid_tracker.mark_finished(miner_uids, MechanismType.ERA5)
        
        bt.logging.info(f"[PROXY] Obtained a valid eager prediction.")
        return self.format_response(
            request_start, 
            prediction, 
            grid, 
            start_time, 
            end_time, 
            converter=variable_conv,
        )


    async def predict_point(self, request: Request):
        self.authorize_token(request.headers)
        bt.logging.info("[PROXY] Received an organic request!")

        request_start = time.time()

        # catch errors to prevent log spam if API is missused
        try:
            payload = await request.json()
            lat = payload["lat"]
            lon = payload["lon"]

            grid = expand_to_grid(lat, lon)

            start_time, end_time, predict_hours = self._parse_time_inputs(payload, MechanismType.ERA5)
            variable_conv = self._parse_era5_variable(payload)

            synapse = TimePredictionSynapse(
                locations=grid.tolist(),
                variable=variable_conv.data_var,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                requested_hours=predict_hours,
            )

        except Exception as e:
            bt.logging.info(f"[PROXY] Organic request was invalid.")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request, parsing failed with error:\n {traceback.format_exc()}",
            )

        # getting responses
        miner_uids = self.get_proxy_uids(MechanismType.ERA5)
        self.validator.uid_tracker.add_busy_uids(miner_uids, MechanismType.ERA5)
        axons = [self.validator.metagraph.axons[uid] for uid in miner_uids]

        prediction = self.safe_average(await self.dendrite(
            axons=axons,
            synapse=synapse,
            timeout=self.PROXY_TIMEOUT,
        ), correct_shape=(predict_hours, grid.shape[0], grid.shape[1]))

        self.validator.uid_tracker.mark_finished(miner_uids, MechanismType.ERA5)
        bt.logging.info(f"[PROXY] Obtained a valid eager prediction.")

        prediction = interp_coordinates(prediction, grid, lat, lon)
        expanded_loc = torch.tensor([lat, lon])[None, None, :]
        return self.format_response(
            request_start, 
            prediction, 
            expanded_loc, 
            start_time, 
            end_time, 
            converter=variable_conv,
        )

    async def predict_station(self, request: Request):
        self.authorize_token(request.headers)
        bt.logging.info("[PROXY] Received an organic request!")

        request_start = time.time()

        # catch errors to prevent log spam if API is missused
        try:
            payload = await request.json()
            station_id = payload["station_id"]
            try:
                station: Station = self.validator.data_loaders[MechanismType.WEATHER_XM].stations[station_id]
            except KeyError:
                raise HTTPException(status_code=400, detail="Station not found")
            
            start_time, end_time, predict_hours = self._parse_time_inputs(payload, MechanismType.WEATHER_XM)
            variable_conv = self._parse_weatherxm_variable(payload)

            synapse = LocalPredictionSynapse(
                latitude=station.latitude,
                longitude=station.longitude,
                elevation=station.elevation,
                variable=variable_conv.data_var,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                requested_hours=predict_hours,
            )

        except Exception as e:
            bt.logging.info(f"[PROXY] Organic request was invalid.")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request, parsing failed with error:\n {traceback.format_exc()}",
            )

        # getting responses
        miner_uids = self.get_proxy_uids(MechanismType.WEATHER_XM)
        self.validator.uid_tracker.add_busy_uids(miner_uids, MechanismType.WEATHER_XM)
        axons = [self.validator.metagraph.axons[uid] for uid in miner_uids]

        prediction = self.safe_average(await self.dendrite(
            axons=axons,
            synapse=synapse,
            timeout=self.PROXY_TIMEOUT,
        ), correct_shape=(predict_hours, ))

        self.validator.uid_tracker.mark_finished(miner_uids, MechanismType.WEATHER_XM)
        bt.logging.info(f"[PROXY] Obtained a valid eager prediction.")

        expanded_loc = torch.tensor([station.latitude, station.longitude])[None, None, :]
        return self.format_response(
            request_start, 
            prediction, 
            expanded_loc, 
            start_time, 
            end_time, 
            converter=variable_conv,
        )


    def _parse_time_inputs(self, payload, mechanism: MechanismType):
        start_time = payload.get("start_time", get_today("h") + pd.Timedelta(hours=1))
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time).floor("h").replace(tzinfo=None)

        predict_hours = payload.get("predict_hours", None)

        end_time = payload.get("end_time", start_time + pd.Timedelta(hours=(predict_hours or 24) - 1))
        if isinstance(end_time, str):
           end_time = pd.Timestamp(end_time).floor("h").replace(tzinfo=None)

        if predict_hours is None:
            predict_hours = get_hours(start_time, end_time) + 1

        assert (
            (end_time - start_time) / pd.Timedelta(hours=1) + 1 == predict_hours
        ), "The difference between start and end timestamps does not match predict_hours."

        assert (
            isinstance(predict_hours, int) and predict_hours > 0 and predict_hours <= 24 
        ), "Prediction hours needs to be an integer between 1 and 24."

        start_offset = get_hours(get_today("h"), start_time)
        if mechanism == MechanismType.ERA5:
            assert (
                start_offset >= ERA5_START_OFFSET_RANGE[0] and start_offset < ERA5_START_OFFSET_RANGE[1]
            ), "You start time can only be between 5 days in the past up to 7 days in the future"
        elif mechanism == MechanismType.WEATHER_XM:
            assert (
                start_offset >= WEATHERXM_START_OFFSET_RANGE[0] and start_offset < WEATHERXM_START_OFFSET_RANGE[1]
            ), "You start time can only one hour into the future up to 6 days in the future"
    
        return start_time, end_time, int(predict_hours)
    
    def _parse_era5_variable(self, payload) -> VariableConverter:
        # default to 2m_temperature for backwards compatibility
        variable_name = payload.get("variable", "2m_temperature")
        assert variable_name in ERA5_DATA_VARS, "This variable is not supported yet!"
        return get_converter(variable_name)

    def _parse_weatherxm_variable(self, payload) -> WeatherXMVariable:
        variable_name = payload.get("variable", "temperature")
        assert variable_name in WEATHER_XM_DATA_VARS, "This variable is not supported yet!"
        return get_wxm_variable(variable_name)

    def format_response(
            self, 
            generation_start: float, 
            prediction: torch.Tensor, 
            location_grid: torch.Tensor,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp,
            converter: Union[VariableConverter, WeatherXMVariable],
    ) -> Dict[str, Any]:
        
        timestamps = pd.date_range(
            start_time.tz_localize("GMT+0"), 
            end_time.tz_localize("GMT+0"),
            freq="h"
        )
        
        tz_info = [
            [
                self.timezone_finder.timezone_at(lng=lon, lat=lat)
                for lat, lon in lat_row
            ] 
            for lat_row in location_grid
        ]

        time_data = [
            [
                [
                    str(safe_tz_convert(timestamp, tz))
                    for tz in row
                ] 
                for row in tz_info
            ]
            for timestamp in timestamps
        ]

        return {
                "generation_time": time.time() - generation_start,
                "grid": location_grid.tolist(),
                converter.data_var: {
                    "data": prediction.tolist(),
                    "unit": converter.unit
                },
                "time": {
                    "data": time_data,
                    "unit": "ISO 8601 (tz-aware)"
                }
            }
    
def is_valid_response(response: torch.Tensor, correct_shape: Union[torch.Size, Tuple[int]]) -> bool:
    if not isinstance(correct_shape, torch.Size):
        correct_shape = torch.Size(correct_shape)
    prediction = help_format_miner_output(correct_shape, response)
    return not get_shape_penalty(correct_shape, prediction)