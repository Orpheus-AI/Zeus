"""
Unit tests for zeus.utils.time module.
"""
import pytest
import pandas as pd
from unittest.mock import patch
from zeus.utils.time import (
    timestamp_to_str,
    get_today,
    get_hours,
    safe_tz_convert,
    to_timestamp,
)


class TestTimestampToStr:
    """Tests for timestamp_to_str function."""

    def test_timestamp_to_str(self):
        """Test converting timestamp to string."""
        timestamp = 1000000.0
        result = timestamp_to_str(timestamp)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_timestamp_to_str_format(self):
        """Test that timestamp string has correct format."""
        timestamp = 1000000.0
        result = timestamp_to_str(timestamp)
        # Should be in format "YYYY-MM-DD HH:MM:SS"
        parts = result.split()
        assert len(parts) == 2
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3


class TestGetToday:
    """Tests for get_today function."""

    def test_get_today_returns_timestamp(self):
        """Test that get_today returns a Timestamp."""
        result = get_today()
        assert isinstance(result, pd.Timestamp)

    def test_get_today_with_floor(self):
        """Test get_today with floor parameter."""
        result = get_today(floor="h")
        assert isinstance(result, pd.Timestamp)

    def test_get_today_timezone_naive(self):
        """Test that get_today returns timezone-naive timestamp."""
        result = get_today()
        assert result.tz is None


class TestGetHours:
    """Tests for get_hours function."""

    def test_get_hours_basic(self):
        """Test basic hours calculation."""
        start = pd.Timestamp("2024-01-01 00:00:00")
        # Use next day at 00:00:00 instead of 24:00:00 (which is invalid)
        end = pd.Timestamp("2024-01-02 00:00:00")
        result = get_hours(start, end)
        assert result == 24

    def test_get_hours_partial(self):
        """Test partial hours calculation."""
        start = pd.Timestamp("2024-01-01 00:00:00")
        end = pd.Timestamp("2024-01-01 12:30:00")
        result = get_hours(start, end)
        assert result == 12

    def test_get_hours_zero(self):
        """Test zero hours."""
        start = pd.Timestamp("2024-01-01 00:00:00")
        end = pd.Timestamp("2024-01-01 00:00:00")
        result = get_hours(start, end)
        assert result == 0


class TestSafeTZConvert:
    """Tests for safe_tz_convert function."""

    def test_safe_tz_convert_with_tz(self):
        """Test converting timestamp with timezone."""
        timestamp = pd.Timestamp("2024-01-01 00:00:00", tz="GMT+0")
        result = safe_tz_convert(timestamp, "America/New_York")
        assert isinstance(result, pd.Timestamp)
        assert result.tz is not None

    def test_safe_tz_convert_without_tz(self):
        """Test converting timestamp without timezone."""
        timestamp = pd.Timestamp("2024-01-01 00:00:00")
        result = safe_tz_convert(timestamp, "America/New_York")
        assert isinstance(result, pd.Timestamp)

    def test_safe_tz_convert_invalid_tz(self):
        """Test converting with invalid timezone."""
        timestamp = pd.Timestamp("2024-01-01 00:00:00")
        # Should return original timestamp on error
        result = safe_tz_convert(timestamp, "Invalid/Timezone")
        assert isinstance(result, pd.Timestamp)


class TestToTimestamp:
    """Tests for to_timestamp function."""

    def test_to_timestamp_basic(self):
        """Test basic timestamp conversion."""
        float_ts = 1000000.0
        result = to_timestamp(float_ts)
        assert isinstance(result, pd.Timestamp)

    def test_to_timestamp_timezone_naive(self):
        """Test that to_timestamp returns timezone-naive timestamp."""
        float_ts = 1000000.0
        result = to_timestamp(float_ts)
        assert result.tz is None

    def test_to_timestamp_roundtrip(self):
        """Test roundtrip conversion."""
        float_ts = 1000000.0
        timestamp = to_timestamp(float_ts)
        # Convert back to float (approximate)
        back_to_float = timestamp.timestamp()
        assert abs(back_to_float - float_ts) < 1.0  # Within 1 second

