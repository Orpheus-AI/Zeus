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

from typing import Optional
import pandas as pd
import pytz


def timestamp_to_str(float_timestamp: float) -> str:
    """
    Convert a float timestamp to a string in the format of YYYY-MM-DD HH:MM:SS.
    Args:
        float_timestamp: The float timestamp to convert.

    Returns:
        str: The string timestamp.
    """
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
    """
    Get the number of hours between two timestamps, only including one boundary.
    Args:
        start: The start timestamp.
        end: The end timestamp.

    Returns:
        int: The number of hours between the two timestamps.
    """
    return int((end - start) / pd.Timedelta(hours=1))

def safe_tz_convert(timestamp: pd.Timestamp, tz: str) -> pd.Timestamp:
    """
    Convert a timestamp to a given timezone,
    if the timestamp is timezone naive, we localize it to GMT+0 first before converting.
    Args:
        timestamp: The timestamp to convert.
        tz: The timezone to convert to.

    Returns:
        pd.Timestamp: The converted timestamp.
    """
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