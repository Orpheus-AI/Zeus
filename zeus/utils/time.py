from typing import Optional
import pandas as pd
import pytz


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


def to_timestamp(float_timestamp: float) -> pd.Timestamp:
    """
    Convert a float timestamp (used for storage) to a pandas timestamp, considering that Copernicus is inside GMT+0.
    We strip off the timezone information to make it TZ-naive again (but according to Copernicus' time).
    """
    return pd.Timestamp(float_timestamp, unit="s", tz="GMT+0").replace(tzinfo=None)