"""
Unit tests for zeus.validator.constants module.
"""
import pytest
from pathlib import Path
from zeus.validator.constants import (
    MechanismType,
    FORWARD_DELAY_SECONDS,
    DATABASE_LOCATION,
    HOURS_PREDICT_RANGE,
    PREFERENCE_UPDATE_FREQUENCY,
    REWARD_DIFFICULTY_SCALER,
    REWARD_RMSE_WEIGHT,
    REWARD_EFFICIENCY_WEIGHT,
    MIN_RELATIVE_SCORE,
    MAX_RELATIVE_SCORE,
    CAP_FACTOR_EFFICIENCY,
    EFFICIENCY_THRESHOLD,
    ERA5_DATA_VARS,
    ERA5_LATITUDE_RANGE,
    ERA5_LONGITUDE_RANGE,
    ERA5_AREA_SAMPLE_RANGE,
    WEATHER_XM_DATA_VARS,
    MECHAGRAPH_SIZES,
)


class TestMechanismType:
    """Tests for MechanismType enum."""

    def test_mechanism_type_values(self):
        """Test that MechanismType has correct values."""
        assert MechanismType.ERA5.value == 0
        assert MechanismType.WEATHER_XM.value == 1

    def test_mechanism_type_names(self):
        """Test that MechanismType has correct names."""
        assert MechanismType.ERA5.name == "ERA5"
        assert MechanismType.WEATHER_XM.name == "WEATHER_XM"


class TestConstants:
    """Tests for various constants."""

    def test_forward_delay_seconds(self):
        """Test FORWARD_DELAY_SECONDS constant."""
        assert isinstance(FORWARD_DELAY_SECONDS, (int, float))
        assert FORWARD_DELAY_SECONDS >= 0

    def test_database_location(self):
        """Test DATABASE_LOCATION constant."""
        assert isinstance(DATABASE_LOCATION, Path)

    def test_hours_predict_range(self):
        """Test HOURS_PREDICT_RANGE constant."""
        assert isinstance(HOURS_PREDICT_RANGE, tuple)
        assert len(HOURS_PREDICT_RANGE) == 2
        assert HOURS_PREDICT_RANGE[0] < HOURS_PREDICT_RANGE[1]

    def test_preference_update_frequency(self):
        """Test PREFERENCE_UPDATE_FREQUENCY constant."""
        assert isinstance(PREFERENCE_UPDATE_FREQUENCY, int)
        assert PREFERENCE_UPDATE_FREQUENCY > 0

    def test_reward_constants(self):
        """Test reward-related constants."""
        assert REWARD_DIFFICULTY_SCALER >= 1.0
        assert 0 <= REWARD_RMSE_WEIGHT <= 1.0
        assert 0 <= REWARD_EFFICIENCY_WEIGHT <= 1.0
        assert abs(REWARD_RMSE_WEIGHT + REWARD_EFFICIENCY_WEIGHT - 1.0) < 0.01
        assert MIN_RELATIVE_SCORE <= MAX_RELATIVE_SCORE
        assert CAP_FACTOR_EFFICIENCY > 0
        assert EFFICIENCY_THRESHOLD >= 0

    def test_era5_data_vars(self):
        """Test ERA5_DATA_VARS constant."""
        assert isinstance(ERA5_DATA_VARS, dict)
        assert len(ERA5_DATA_VARS) > 0
        # Check that weights sum to approximately 1.0
        total_weight = sum(ERA5_DATA_VARS.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_era5_ranges(self):
        """Test ERA5 range constants."""
        assert ERA5_LATITUDE_RANGE[0] == -90.0
        assert ERA5_LATITUDE_RANGE[1] == 90.0
        assert ERA5_LONGITUDE_RANGE[0] == -180.0
        assert ERA5_LONGITUDE_RANGE[1] <= 180.0
        assert ERA5_AREA_SAMPLE_RANGE[0] < ERA5_AREA_SAMPLE_RANGE[1]

    def test_weather_xm_data_vars(self):
        """Test WEATHER_XM_DATA_VARS constant."""
        assert isinstance(WEATHER_XM_DATA_VARS, dict)
        assert len(WEATHER_XM_DATA_VARS) > 0
        # Check that weights sum to approximately 1.0
        total_weight = sum(WEATHER_XM_DATA_VARS.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_mechagraph_sizes(self):
        """Test MECHAGRAPH_SIZES constant."""
        assert isinstance(MECHAGRAPH_SIZES, dict)
        assert MechanismType.ERA5 in MECHAGRAPH_SIZES
        assert MechanismType.WEATHER_XM in MECHAGRAPH_SIZES
        assert all(size > 0 for size in MECHAGRAPH_SIZES.values())

