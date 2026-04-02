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
    attempts_per_miner=2,
    max_response_body_bytes=1024 * 1024 * 10,
    forward_timeout=13.0,
)
SHORT_CHALLENGE = (0, 48)
LONG_CHALLENGE = (0, 24 * 15)
# Per-window prediction dendrite settings keyed by (start_offset, end_offset)
PREDICTION_SETTINGS_PER_WINDOW: Dict[Tuple[int, int], DendriteSettings] = {
    SHORT_CHALLENGE: DendriteSettings(
        forward_concurrency=13,
        response_batch_k=26,
        attempts_per_miner=2,
        max_response_body_bytes=1024 * 1024 * 140,
        forward_timeout=26,
    ),
    LONG_CHALLENGE: DendriteSettings(
        forward_concurrency=2,
        response_batch_k=2,
        attempts_per_miner=2,
        max_response_body_bytes=1024 * 1024 * 780,
        forward_timeout=55.0,
    ),
}

# after how many percent of above it yields results
RANK_HISTORY_PRUNE_LEN = 1000 # how many ranks to keep in history for each hotkey after that we prune note that this number can be larger than window size used for ranking

# the corresponding ERA5 variables miners are tested on with their scoring weight
ERA5_DATA_VARS: Dict[str, float] = {
    "2m_temperature": 0.2, 
    "100m_u_component_of_wind": 0.3,
    "100m_v_component_of_wind": 0.3,
    "surface_solar_radiation_downwards": 0.2
}
ERA5_LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
ERA5_LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 179.75)  # real ERA5 ranges
ERA5_RESOLUTION = 0.25
# how many datapoints we want. The resolution is 0.25 degrees, so 4 means 1 degree.
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (4, 16) # 

# Axis-aligned bounding box for Europe (lat °N, lon °E)
EUROPE_LATITUDE_RANGE = (34.0, 72.0)
EUROPE_LONGITUDE_RANGE = (-25.0, 45.0)
EUROPE_WEIGHT = 1.5
# ------------------------------------------------------
# --------------- Current/Future prediction-------------
# ------------------------------------------------------
CURRENT_DIRECTORY: Path = Path.home()

BEST_FORECASTS_DIRECTORY: Path = CURRENT_DIRECTORY  / "Zeus" / "best_prediction" 
ERA5_CACHE_DIR: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "era5"
OLD_METADATA_DATABASE_LOCATION: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "challenges.db"
METADATA_DATABASE_LOCATION: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "challenges_v2.db"
LATITUDE_WEIGHTS_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "weights"
    / "latitude_weights_for_rmse.npy"
) 
COPERNICUS_ERA5_URL: str = "https://cds.climate.copernicus.eu/api"

DEFAULT_STEP_SIZE: int = 1  # hours between prediction time steps (synapse default)
MIN_HOURS_BETWEEN_REQUESTS = 5

TIME_WINDOWS_PER_CHALLENGE: List[Tuple[int, int]] = [SHORT_CHALLENGE, LONG_CHALLENGE]

PERCENTAGE_GOING_TO_WINNER = 0.95

PERFORMANCE_DATABASE_URL = "https://testnet.zeussubnet.com"

# ---- Challenge registry (variable × time-window, each with its own state_key) ----
from zeus.validator.challenge_spec import build_challenge_registry, ChallengeSpec  # noqa: E402

CHALLENGE_REGISTRY: Dict[str, ChallengeSpec] = build_challenge_registry(
    era5_data_vars=ERA5_DATA_VARS,
    time_windows=TIME_WINDOWS_PER_CHALLENGE,
    prediction_settings_per_window=PREDICTION_SETTINGS_PER_WINDOW,
)
