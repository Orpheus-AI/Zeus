from datetime import datetime, timedelta
from typing import Optional
from zeus.validator.constants import CHALLENGE_HASHING_MAX_MINUTE, COMMITMENT_MAX_BLOCKS_OLDER

import pandas as pd
import pytz
import os
from typing import Tuple

_TEST_MODE = os.environ.get("ZEUS_TEST_MODE", "0") == "1"
_TEST_STALE_MAX_BLOCKS = int(os.environ.get("ZEUS_TEST_STALE_MAX_BLOCKS", "100"))


def timestamp_to_str(float_timestamp: float) -> str:
    return to_timestamp(float_timestamp).strftime("%Y-%m-%d %H:%M:%S")

def get_today(floor: Optional[str] = None) -> pd.Timestamp:
    """
    Copernicus is inside GMT+0, so we can always use that timezone to get the current day and hour matching theirs.
    But then remove the timezone information so we can actually compare with the dataset (which is TZ-naive).
    """

    timestamp = pd.Timestamp.now(tz="GMT+0").replace(tzinfo=None)
    if floor:
        return timestamp.floor(floor)
    return timestamp

def get_hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start) / pd.Timedelta(hours=1))

def safe_tz_convert(timestamp: pd.Timestamp, tz: str):
    if not timestamp.tz:
        timestamp = timestamp.tz_localize("GMT+0")
    try:
        return timestamp.tz_convert(pytz.timezone(tz))
    except:
        return timestamp


def estimate_block_at_target_time(
    target_time: datetime,
    current_block: int,
    current_time: datetime,
    block_time_sec: int = 12,
) -> int:
    """Estimate block number at a given target time."""
    delta = (target_time - current_time).total_seconds()
    block_delta = int(delta / block_time_sec)
    return current_block + block_delta

def get_challenge_time(now: datetime) -> datetime:
    return now.replace(minute=CHALLENGE_HASHING_MAX_MINUTE, second=0, microsecond=0)

def next_challenge_time(now: datetime) -> datetime:
    """Next :45 on 0/6/12/18 UTC (this cycle if still before :45, else next)."""
    cycle_hour = (now.hour // 6) * 6
    deadline = now.replace(
        hour=cycle_hour, minute=CHALLENGE_HASHING_MAX_MINUTE, second=0, microsecond=0
    )
    if now < deadline:
        return deadline
    nxt = cycle_hour + 6
    if nxt >= 24:
        return (deadline + timedelta(days=1)).replace(
            hour=0, minute=CHALLENGE_HASHING_MAX_MINUTE, second=0, microsecond=0
        )
    return deadline.replace(hour=nxt)


def to_timestamp(float_timestamp: float) -> pd.Timestamp:
    """
    Convert a float timestamp (used for storage) to a pandas timestamp, considering that Copernicus is inside GMT+0.
    We strip off the timezone information to make it TZ-naive again (but according to Copernicus' time).
    """
    return pd.Timestamp(float_timestamp, unit="s", tz="GMT+0").replace(tzinfo=None)


def _sample_commit_wall(sample):
    """UTC wall time at CHALLENGE_HASHING_MAX_MINUTE on the sample's 6h cycle."""
    cycle_base = (
        pd.Timestamp(sample.start_timestamp, unit="s", tz="UTC")
        .floor("6h")
        .tz_convert("GMT+0")
        #.replace(tzinfo=None)
        .to_pydatetime()
    )
    return get_challenge_time(cycle_base)


def _block_parameters_for_sample(
    sample,
    chain_head_block: int,
    chain_head_time,
) -> Tuple[int, int | None, int, int]:
    """Return commit_block, read_block, reference_block, and staleness window for one sample."""
    commit_wall = _sample_commit_wall(sample)
    commit_block = estimate_block_at_target_time(
        commit_wall, chain_head_block, chain_head_time
    )
    read_block = None if _TEST_MODE else commit_block
    reference_block = chain_head_block if _TEST_MODE else commit_block
    max_blocks_older = _TEST_STALE_MAX_BLOCKS if _TEST_MODE else COMMITMENT_MAX_BLOCKS_OLDER
    return commit_block, read_block, reference_block, max_blocks_older