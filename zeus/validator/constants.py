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

from typing import List, Tuple, Dict
from enum import IntEnum, unique
from pathlib import Path

# ------------------------------------------------------
# ------------------ General Constants -----------------
# ------------------------------------------------------
TESTNET_UID = 301
MAINNET_UID = 18

FORWARD_DELAY_SECONDS = 90

# how many miners proxy queries
PROXY_QUERY_K = 3

# minimum tao for a validator
VPERMIT_TAO_LIMIT = 4096

# wandb website refuses to update logs after roughly 100k, so reset run if this happens
WANDB_MAX_LOGS = 95_000

DATABASE_LOCATION: Path = Path.home() / ".cache" / "zeus" / "challenges.db"
HOURS_PREDICT_RANGE: Tuple[float, float] = (1, 25) # how many hours ahead we want to predict.

# ------------------------------------------------------
# --------------- Mechanism Constants -----------------
# ------------------------------------------------------
# every how many blocks (roughly) do we update the preference
PREFERENCE_UPDATE_FREQUENCY = 25

@unique
class MechanismType(IntEnum):
    ERA5 = 0
    WEATHER_XM = 1


MECHAGRAPH_SIZES: Dict[MechanismType, int] = {
    MechanismType.ERA5: 128,
    MechanismType.WEATHER_XM: 128,
}
MECHANISM_PROBABILITIES: Dict[MechanismType, float] = {
    MechanismType.ERA5: 0.75,
    MechanismType.WEATHER_XM: 0.25, # so we definitely stay under 20k requests a month
}


# ------------------------------------------------------
# ------------------ Reward Constants -----------------
# ------------------------------------------------------
# 1.0 would imply no difficulty scaling, should be >= 1.
REWARD_DIFFICULTY_SCALER = 2.0
# 70% of emission for quality, 30% for speed
REWARD_RMSE_WEIGHT = 0.8
REWARD_EFFICIENCY_WEIGHT = 0.2
# score is percentage worse/better than OpenMeteo baseline. Capped between these percentages (as float)
MIN_RELATIVE_SCORE = -1.0
MAX_RELATIVE_SCORE = 0.8
# when curving scores, above cap * median_speed = 0
# to prevent reward curve from being shifted by really bad outlier
CAP_FACTOR_EFFICIENCY = 2.0
# Faster than this is considered 'perfect'
EFFICIENCY_THRESHOLD = 0.4

# ------------------------------------------------------
# ------------------- ERA5 predictions -----------------
# ------------------------------------------------------
# the variables miners are tested on, with their respective sampling weight
ERA5_DATA_VARS: Dict[str, float] = {
    "2m_temperature": 0.15,
    "total_precipitation": 0.15,
    "100m_u_component_of_wind": 0.2,
    "100m_v_component_of_wind": 0.2,
    "2m_dewpoint_temperature": 0.2,
    "surface_pressure": 0.1
}
ERA5_LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
ERA5_LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 179.75)  # real ERA5 ranges
# how many datapoints we want. The resolution is 0.25 degrees, so 4 means 1 degree.
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (4, 16)

ERA5_CACHE_DIR: Path = Path.home() / ".cache" / "zeus" / "era5"
COPERNICUS_ERA5_URL: str = "https://cds.climate.copernicus.eu/api"

ERA5_START_OFFSET_RANGE: Tuple[int, int] = (-119, 168)  # 4 days and 23 hours ago <---> until 7 days in future
ERA5_UNIFORM_START_OFFSET_PROB: float = 0.1

# see plot of distribution in Zeus/static/era5_start_offset_distribution.png
ERA5_START_SAMPLE_STD: float = 35 

OPEN_METEO_URL: str = "https://customer-api.open-meteo.com/v1/forecast"

# ------------------------------------------------------
# ---------------- WeatherXM predictions ---------------
# ------------------------------------------------------
WEATHER_XM_URL = "https://pro.weatherxm.com/api/v1"
WEATHER_XM_DATA_VARS: Dict[str, float] = {
    "temperature": 0.3,
    "wind_speed": 0.15,
    "wind_direction": 0.15,
    "humidity": 0.2,
    "pressure": 0.2
}

# start hour offset is sampled from a folded normal with mean zero and below std
WEATHERXM_START_SAMPLE_STD: float = 35
WEATHERXM_START_OFFSET_RANGE: Tuple[int, int] = (1, 120)
WEATHERXM_MIN_DATA_QUALITY: float = 0.95

WEATHERXM_CELL_RESOLUTION: int = 7  # h3 cell resolution used by WeatherXM

# ------------------------------------------------------
# ---------- Historic prediction (UNUSED) --------------
# ------------------------------------------------------
# ERA5 data loading constants
GCLOUD_ERA5_URL: str = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

HISTORIC_INPUT_HOURS: int = 120 # How many hours of data we send miners
HISTORIC_HOURS_PREDICT_RANGE: Tuple[float, float] = (1, 9) # how many hours ahead we want to predict.
HISTORIC_DATE_RANGE: Tuple[str, str] = (
    "1960-01-01",
    "2024-10-31",
)  # current latest inside that Zarr archive

MIN_INTERPOLATION_DISTORTIONS = 5
MAX_INTERPOLATION_DISTORTIONS = 50

