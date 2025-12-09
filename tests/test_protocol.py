"""
Unit tests for zeus.protocol module.
"""
import pytest
import torch
from zeus.protocol import (
    PreferenceSynapse,
    PredictionSynapse,
    TimePredictionSynapse,
    LocalPredictionSynapse,
)
from zeus.validator.constants import MechanismType


class TestPreferenceSynapse:
    """Tests for PreferenceSynapse class."""

    def test_init_default(self):
        """Test default initialization."""
        synapse = PreferenceSynapse()
        assert synapse.mechanism is None

    def test_init_with_mechanism(self):
        """Test initialization with mechanism."""
        synapse = PreferenceSynapse(mechanism=MechanismType.ERA5)
        assert synapse.mechanism == MechanismType.ERA5

    def test_deserialize(self):
        """Test deserialize method."""
        synapse = PreferenceSynapse(mechanism=MechanismType.WEATHER_XM)
        result = synapse.deserialize()
        assert result == MechanismType.WEATHER_XM


class TestPredictionSynapse:
    """Tests for PredictionSynapse class.
    
    Note: PredictionSynapse is an abstract base class and doesn't define
    the predictions field - that's done in subclasses (TimePredictionSynapse, LocalPredictionSynapse).
    """

    def test_init_default(self):
        """Test default initialization."""
        # Note: Can't instantiate ABC directly, but we can test via subclass
        # This test verifies the base class fields exist in subclasses
        synapse = TimePredictionSynapse()
        assert synapse.version == ""
        assert synapse.requested_hours == 1
        assert synapse.start_time == 0.0
        assert synapse.end_time == 0.0

    def test_init_with_values(self):
        """Test initialization with values."""
        synapse = TimePredictionSynapse(
            version="2.0.0",
            requested_hours=24,
            start_time=1000000.0,
            end_time=1000024.0,
        )
        assert synapse.version == "2.0.0"
        assert synapse.requested_hours == 24
        assert synapse.start_time == 1000000.0
        assert synapse.end_time == 1000024.0


class TestTimePredictionSynapse:
    """Tests for TimePredictionSynapse class."""

    def test_init_default(self):
        """Test default initialization."""
        synapse = TimePredictionSynapse()
        assert synapse.locations == []
        assert synapse.predictions == []
        assert synapse.variable == "2m_temperature"

    def test_init_with_values(self):
        """Test initialization with values."""
        locations = [[(10.0, 20.0), (11.0, 21.0)]]
        predictions = [[[1.0, 2.0], [3.0, 4.0]]]
        synapse = TimePredictionSynapse(
            locations=locations,
            predictions=predictions,
            variable="total_precipitation"
        )
        assert synapse.locations == locations
        assert synapse.predictions == predictions
        assert synapse.variable == "total_precipitation"

    def test_deserialize(self):
        """Test deserialize method for TimePredictionSynapse."""
        predictions = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        synapse = TimePredictionSynapse(predictions=predictions)
        result = synapse.deserialize()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2, 2)


class TestLocalPredictionSynapse:
    """Tests for LocalPredictionSynapse class."""

    def test_init_default(self):
        """Test default initialization."""
        synapse = LocalPredictionSynapse()
        assert synapse.latitude == 0.0
        assert synapse.longitude == 0.0
        assert synapse.elevation == 0.0
        assert synapse.predictions == []
        assert synapse.variable == "temperature"

    def test_init_with_values(self):
        """Test initialization with values."""
        synapse = LocalPredictionSynapse(
            latitude=40.7128,
            longitude=-74.0060,
            elevation=10.5,
            predictions=[1.0, 2.0, 3.0],
            variable="humidity"
        )
        assert synapse.latitude == 40.7128
        assert synapse.longitude == -74.0060
        assert synapse.elevation == 10.5
        assert synapse.predictions == [1.0, 2.0, 3.0]
        assert synapse.variable == "humidity"

    def test_deserialize(self):
        """Test deserialize method."""
        predictions = [1.0, 2.0, 3.0, 4.0]
        synapse = LocalPredictionSynapse(predictions=predictions)
        result = synapse.deserialize()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)

