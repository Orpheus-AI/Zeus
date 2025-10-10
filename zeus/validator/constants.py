from typing import List, Tuple, Dict
from pathlib import Path

# ------------------------------------------------------
# ------------------ General Constants -----------------
# ------------------------------------------------------
TESTNET_UID = 301
MAINNET_UID = 18

FORWARD_DELAY_SECONDS = 90

# how many miners proxy queries
PROXY_QUERY_K = 10
# after how many percent of above it yields results
PROXY_CUTOFF_PERCENT = 0.6

# wandb website refuses to update logs after roughly 100k, so reset run if this happens
WANDB_MAX_LOGS = 95_000

# the variables miners are tested on, with their respective sampling weight
ERA5_DATA_VARS: Dict[str, float] = {
    "2m_temperature": 0.2,
    "total_precipitation": 0.2,
    "100m_u_component_of_wind": 0.2,
    "100m_v_component_of_wind": 0.2,
    "2m_dewpoint_temperature": 0.2,
}
ERA5_LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
ERA5_LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 179.75)  # real ERA5 ranges
# how many datapoints we want. The resolution is 0.25 degrees, so 4 means 1 degree.
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (4, 16)

# ------------------------------------------------------
# ------------------ Reward Constants -----------------
# ------------------------------------------------------
# 1.0 would imply no difficulty scaling, should be >= 1.
REWARD_DIFFICULTY_SCALER = 3.0
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
# --------------- Current/Future prediction-------------
# ------------------------------------------------------
ERA5_CACHE_DIR: Path = Path.home() / ".cache" / "zeus" / "era5"
DATABASE_LOCATION: Path = Path.home() / ".cache" / "zeus" / "challenges.db"
COPERNICUS_ERA5_URL: str = "https://cds.climate.copernicus.eu/api"

LIVE_START_OFFSET_RANGE: Tuple[int, int] = (-119, 168)  # 4 days and 23 hours ago <---> until 7 days in future
LIVE_UNIFORM_START_OFFSET_PROB: float = 0.1
LIVE_HOURS_PREDICT_RANGE: Tuple[float, float] = (1, 25) # how many hours ahead we want to predict.

# see plot of distribution in Zeus/static/era5_start_offset_distribution.png
LIVE_START_SAMPLE_STD: float = 35 

# ------------------------------------------------------
# ------------ OpenMeteo (SOTA comparisons) ------------
# ------------------------------------------------------
OPEN_METEO_URL: str = "https://customer-api.open-meteo.com/v1/forecast"

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

