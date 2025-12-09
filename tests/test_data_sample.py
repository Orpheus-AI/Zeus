"""
Unit tests for zeus.data sample classes.
"""
import pytest
import torch
from unittest.mock import Mock, patch
from zeus.data.base.sample import BaseSample
from zeus.data.era5.sample import Era5Sample
from zeus.data.weatherxm.sample import WeatherXMSample
from zeus.protocol import TimePredictionSynapse, LocalPredictionSynapse
from zeus.validator.constants import MechanismType


class TestBaseSample:
    """Tests for BaseSample abstract class."""

    def test_base_sample_init_with_output_data(self):
        """Test BaseSample initialization with output_data."""
        output_data = torch.randn(24, 5, 5)
        
        class ConcreteSample(BaseSample):
            def get_synapse(self):
                pass
            
            def get_bbox(self):
                return (-10.0, 10.0, -20.0, 20.0)
            
            @property
            def mechanism(self):
                return MechanismType.ERA5
        
        sample = ConcreteSample(
            start_timestamp=1000000.0,
            end_timestamp=1000864.0,
            variable="2m_temperature",
            output_data=output_data
        )
        
        assert sample.hours_to_predict == 24
        assert torch.equal(sample.output_data, output_data)

    def test_base_sample_init_with_hours_to_predict(self):
        """Test BaseSample initialization with hours_to_predict."""
        class ConcreteSample(BaseSample):
            def get_synapse(self):
                pass
            
            def get_bbox(self):
                return (-10.0, 10.0, -20.0, 20.0)
            
            @property
            def mechanism(self):
                return MechanismType.ERA5
        
        sample = ConcreteSample(
            start_timestamp=1000000.0,
            end_timestamp=1000864.0,
            variable="2m_temperature",
            hours_to_predict=24
        )
        
        assert sample.hours_to_predict == 24
        assert sample.output_data is None

    def test_base_sample_init_requires_output_or_hours(self):
        """Test that BaseSample requires either output_data or hours_to_predict."""
        class ConcreteSample(BaseSample):
            def get_synapse(self):
                pass
            
            def get_bbox(self):
                return (-10.0, 10.0, -20.0, 20.0)
            
            @property
            def mechanism(self):
                return MechanismType.ERA5
        
        with pytest.raises(ValueError):
            ConcreteSample(
                start_timestamp=1000000.0,
                end_timestamp=1000864.0,
                variable="2m_temperature"
            )

    def test_base_sample_inserted_at_defaults_to_current_time(self):
        """Test that inserted_at defaults to current time."""
        import time
        
        class ConcreteSample(BaseSample):
            def get_synapse(self):
                pass
            
            def get_bbox(self):
                return (-10.0, 10.0, -20.0, 20.0)
            
            @property
            def mechanism(self):
                return MechanismType.ERA5
        
        before = time.time()
        sample = ConcreteSample(
            start_timestamp=1000000.0,
            end_timestamp=1000864.0,
            variable="2m_temperature",
            hours_to_predict=24
        )
        after = time.time()
        
        # inserted_at is rounded, so account for rounding (can round up or down)
        # It should be within 1 second of the time range
        assert int(before) - 1 <= sample.inserted_at <= int(after) + 1


class TestEra5Sample:
    """Tests for Era5Sample class."""

    @pytest.fixture
    def era5_sample(self):
        """Create an Era5Sample for testing."""
        return Era5Sample(
            lat_start=-10.0,
            lat_end=10.0,
            lon_start=-20.0,
            lon_end=20.0,
            start_timestamp=1000000.0,
            end_timestamp=1000864.0,
            variable="2m_temperature",
            hours_to_predict=24
        )

    def test_era5_sample_mechanism(self, era5_sample):
        """Test that Era5Sample has correct mechanism."""
        assert era5_sample.mechanism == MechanismType.ERA5

    def test_era5_sample_get_bbox(self, era5_sample):
        """Test that get_bbox returns correct bounding box."""
        bbox = era5_sample.get_bbox()
        assert bbox == (-10.0, 10.0, -20.0, 20.0)

    def test_era5_sample_has_x_grid(self, era5_sample):
        """Test that Era5Sample creates x_grid."""
        assert hasattr(era5_sample, 'x_grid')
        assert era5_sample.x_grid is not None

    def test_era5_sample_get_synapse(self, era5_sample):
        """Test that get_synapse returns correct TimePredictionSynapse."""
        synapse = era5_sample.get_synapse()
        
        assert isinstance(synapse, TimePredictionSynapse)
        assert synapse.start_time == era5_sample.start_timestamp
        assert synapse.end_time == era5_sample.end_timestamp
        assert synapse.requested_hours == era5_sample.hours_to_predict
        assert synapse.variable == era5_sample.variable
        assert synapse.version is not None
        assert len(synapse.locations) > 0


class TestWeatherXMSample:
    """Tests for WeatherXMSample class."""

    @pytest.fixture
    def weatherxm_sample(self):
        """Create a WeatherXMSample for testing."""
        return WeatherXMSample(
            lat=40.7128,
            lon=-74.0060,
            elevation=10.5,
            station_id="test_station",
            start_timestamp=1000000.0,
            end_timestamp=1000864.0,
            variable="temperature",
            hours_to_predict=24
        )

    def test_weatherxm_sample_mechanism(self, weatherxm_sample):
        """Test that WeatherXMSample has correct mechanism."""
        assert weatherxm_sample.mechanism == MechanismType.WEATHER_XM

    def test_weatherxm_sample_get_bbox(self, weatherxm_sample):
        """Test that get_bbox returns correct bounding box (point location)."""
        bbox = weatherxm_sample.get_bbox()
        # For a point location, lat_start == lat_end and lon_start == lon_end
        assert bbox == (40.7128, 40.7128, -74.0060, -74.0060)

    def test_weatherxm_sample_get_synapse(self, weatherxm_sample):
        """Test that get_synapse returns correct LocalPredictionSynapse."""
        synapse = weatherxm_sample.get_synapse()
        
        assert isinstance(synapse, LocalPredictionSynapse)
        assert synapse.latitude == weatherxm_sample.lat
        assert synapse.longitude == weatherxm_sample.lon
        assert synapse.elevation == weatherxm_sample.elevation
        assert synapse.start_time == weatherxm_sample.start_timestamp
        assert synapse.end_time == weatherxm_sample.end_timestamp
        assert synapse.requested_hours == weatherxm_sample.hours_to_predict
        assert synapse.variable == weatherxm_sample.variable
        assert synapse.version is not None

