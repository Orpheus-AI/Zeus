"""
Unit tests for zeus.data.base.loader module.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from zeus.data.base.loader import BaseDataLoader
from zeus.data.base.sample import BaseSample
from zeus.validator.constants import MechanismType


class TestBaseDataLoader:
    """Tests for BaseDataLoader abstract class."""

    @pytest.fixture
    def concrete_loader(self):
        """Create a concrete implementation of BaseDataLoader for testing."""
        class ConcreteLoader(BaseDataLoader):
            @property
            def mechanism(self):
                return MechanismType.ERA5
            
            @property
            def sample_cls(self):
                return Mock
            
            def get_sample(self):
                return Mock(spec=BaseSample)
            
            def get_last_available(self):
                return pd.Timestamp.now()
            
            def get_output(self, sample):
                return None
        
        return ConcreteLoader(
            data_vars={"2m_temperature": 0.5, "total_precipitation": 0.3, "surface_pressure": 0.2},
            predict_sample_range=(1, 24),
            start_offset_range=(-120, 120)
        )

    def test_base_data_loader_init_sorts_data_vars(self, concrete_loader):
        """Test that data_vars are sorted on initialization."""
        # Should be sorted alphabetically
        assert concrete_loader.data_vars == ("2m_temperature", "surface_pressure", "total_precipitation")

    def test_base_data_loader_init_normalizes_probs(self, concrete_loader):
        """Test that data_var_probs are normalized."""
        # Original weights: 0.5, 0.3, 0.2 (sum = 1.0)
        # After sorting: 0.5, 0.2, 0.3
        expected_probs = np.array([0.5, 0.2, 0.3])
        assert np.allclose(concrete_loader.data_var_probs, expected_probs)

    def test_base_data_loader_init_sorts_ranges(self, concrete_loader):
        """Test that ranges are sorted on initialization."""
        assert concrete_loader.predict_sample_range == (1, 24)
        assert concrete_loader.start_offset_range == (-120, 120)

    def test_base_data_loader_init_sorts_reversed_ranges(self):
        """Test that reversed ranges are sorted correctly."""
        class ConcreteLoader(BaseDataLoader):
            @property
            def mechanism(self):
                return MechanismType.ERA5
            
            @property
            def sample_cls(self):
                return Mock
            
            def get_sample(self):
                return Mock(spec=BaseSample)
            
            def get_last_available(self):
                return pd.Timestamp.now()
            
            def get_output(self, sample):
                return None
        
        loader = ConcreteLoader(
            data_vars={"var1": 1.0},
            predict_sample_range=(24, 1),  # Reversed
            start_offset_range=(120, -120)  # Reversed
        )
        
        assert loader.predict_sample_range == (1, 24)
        assert loader.start_offset_range == (-120, 120)

    def test_is_ready_returns_true(self, concrete_loader):
        """Test that is_ready returns True by default."""
        assert concrete_loader.is_ready() is True

    def test_sample_variable_returns_valid_variable(self, concrete_loader):
        """Test that sample_variable returns a valid variable."""
        # Sample multiple times to ensure it returns valid variables
        for _ in range(10):
            var = concrete_loader.sample_variable()
            assert var in concrete_loader.data_vars

    def test_sample_variable_respects_probabilities(self, concrete_loader):
        """Test that sample_variable respects the probability distribution."""
        # Sample many times and check distribution
        samples = [concrete_loader.sample_variable() for _ in range(1000)]
        counts = {var: samples.count(var) for var in concrete_loader.data_vars}
        
        # The most probable variable should be sampled most often
        # "2m_temperature" has weight 0.5, so should be sampled ~50% of the time
        temp_count = counts["2m_temperature"]
        assert temp_count > 400  # Should be around 500, allow some variance

    def test_get_relative_age_past_prediction(self, concrete_loader):
        """Test get_relative_age for past predictions."""
        sample = Mock(spec=BaseSample)
        sample.end_timestamp = 1000000.0
        sample.inserted_at = 1000100.0  # After end_timestamp
        sample.start_timestamp = 999900.0
        
        # Past prediction: age = inserted_at - start_timestamp
        # age = 1000100.0 - 999900.0 = 200 seconds
        # relative_age = age / start_offset_range[0] (in hours)
        # relative_age = 200 / (-120 * 3600) = negative value
        age = concrete_loader.get_relative_age(sample)
        
        # Should be negative (past prediction)
        assert age < 0

    def test_get_relative_age_future_prediction(self, concrete_loader):
        """Test get_relative_age for future predictions."""
        sample = Mock(spec=BaseSample)
        sample.end_timestamp = 1000100.0
        sample.inserted_at = 1000000.0  # Before end_timestamp
        sample.start_timestamp = 999900.0
        
        # Future prediction: age = end_timestamp - inserted_at
        # age = 1000100.0 - 1000000.0 = 100 seconds
        # relative_age = age / (start_offset_range[1] + predict_sample_range[1]) (in hours)
        # relative_age = 100 / ((120 + 24) * 3600)
        age = concrete_loader.get_relative_age(sample)
        
        # Should be positive (future prediction)
        assert age > 0
        assert age <= 1.0  # Should be normalized

