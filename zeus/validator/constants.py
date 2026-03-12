from pathlib import Path
from typing import Dict, List, Tuple

# ------------------------------------------------------
# ------------------ General Constants -----------------
# ------------------------------------------------------
TESTNET_UID = 301
MAINNET_UID = 18

MAX_MINER_RESPONSE_BODY_BYTES = 1024 * 1024 * 107 # 95MB

FORWARD_DELAY_SECONDS = 90
MAX_TIME_OFFSET = 48 # 48 hours
# Forward run_challenge_phase: batch size and max batches when iterating over all available miners
FORWARD_RESPONSE_BATCH_K = 50 # 65
ATTEMPTS_PER_MINER = 2
# how many miners proxy queries
PROXY_QUERY_K = 10
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
ERA5_AREA_SAMPLE_RANGE: Tuple[float, float] = (4, 16) # TODO maybe we can remove this
# ------------------------------------------------------
# --------------- Current/Future prediction-------------
# ------------------------------------------------------
CURRENT_DIRECTORY: Path = Path.home()

ERA5_CACHE_DIR: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "era5"
METADATA_DATABASE_LOCATION: Path = CURRENT_DIRECTORY / ".cache" / "zeus" / "challenges.db"
COPERNICUS_ERA5_URL: str = "https://cds.climate.copernicus.eu/api"

DEFAULT_STEP_SIZE: int = 1  # hours between prediction time steps (synapse default)
TIME_OFFSET_PER_CHALLENGE: int = 48  # each challenge requests 48 hours; two challenges cover full window
MIN_HOURS_BETWEEN_REQUESTS = 5

# Commit/reveal schedule: slots start at 00:00, 06:00, 12:00, 18:00 UTC only (hour % 6 == 0).
# First K blocks in each slot = commit (hash); next M blocks = reveal (full prediction).
BLOCK_TIME_SECONDS = 12
COMMIT_BLOCKS = 5 * 40  # 5 per minute * 30 minutes = 150 blocks
BREAK_BETWEEN_COMMIT_AND_REVEAL = 5 * 10  # 5 per minute * 10 minutes = 50 blocks
REVEAL_BLOCKS = 5 * 30  # 5 per minute * 30 minutes = 150 blocks

PERMITTED_MINER_STRIKES = 3
MAX_TIME_OFFSET: int = 48  # full window steps (cache range); challenges run as two 24-step requests
PERCENTAGE_GOING_TO_WINNER = 0.95


