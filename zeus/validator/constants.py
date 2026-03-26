from pathlib import Path
from typing import Dict, List, Tuple

from zeus.base.dendrite import DendriteSettings

# ------------------------------------------------------
# ------------------ General Constants -----------------
# ------------------------------------------------------
TESTNET_UID = 301
MAINNET_UID = 18

FORWARD_DELAY_SECONDS = 90


# Hash phase: high concurrency, batch size, retries
HASH_DENDRITE_SETTINGS = DendriteSettings(
    forward_concurrency=125,
    response_batch_k=125,
    attempts_per_miner=3,
    max_response_body_bytes=1024 * 1024 * 10,
)
# Prediction phase: lower concurrency, same batch/retry semantics
PREDICTION_DENDRITE_SETTINGS = DendriteSettings(
    forward_concurrency=13,
    response_batch_k=39,
    attempts_per_miner=2,
    max_response_body_bytes=1024 * 1024 * 110,
)

# after how many percent of above it yields results
RANK_HISTORY_PRUNE_LEN = 1000 # how many ranks to keep in history for each hotkey after that we prune note that this number can be larger than window size used for ranking

# the corresponding ERA5 variables miners are tested on with their scoring weight
ERA5_DATA_VARS: Dict[str, float] = {
    "2m_temperature": 0.2, 
    "100m_u_component_of_wind": 0.4,
    "100m_v_component_of_wind": 0.4,
}
ERA5_LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
ERA5_LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 179.75)  # real ERA5 ranges
ERA5_RESOLUTION = 0.25
# how many datapoints we want. The resolution is 0.25 degrees, so 4 means 1 degree.
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (4, 16) # 
# ------------------------------------------------------
# --------------- Current/Future prediction-------------
# ------------------------------------------------------
CURRENT_DIRECTORY: Path = Path.home()

BEST_FORECASTS_DIRECTORY: Path = CURRENT_DIRECTORY  / "Zeus" / "best_prediction" 
ERA5_CACHE_DIR: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "era5"
OLD_METADATA_DATABASE_LOCATION: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "challenges.db"
METADATA_DATABASE_LOCATION: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "challenges_v2.db"
LATITUDE_WEIGHTS_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "weights" / "latitude_weights_for_rmse.npy"
COPERNICUS_ERA5_URL: str = "https://cds.climate.copernicus.eu/api"

DEFAULT_STEP_SIZE: int = 1  # hours between prediction time steps (synapse default)
TIME_OFFSET_PER_CHALLENGE: int = 48  # each challenge requests 48 hours; two challenges cover full window
MIN_HOURS_BETWEEN_REQUESTS = 5


MAX_TIME_OFFSET: int = 48  # full window steps (cache range); challenges run as two 24-step requests
PERCENTAGE_GOING_TO_WINNER = 0.95

PERFORMANCE_DATABASE_URL = "https://performance.zeussubnet.com"
