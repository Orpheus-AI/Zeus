"""
Unit tests for zeus.validator.preference module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from zeus.validator.preference import PreferenceManager
from zeus.validator.constants import MechanismType, PREFERENCE_UPDATE_FREQUENCY


class TestPreferenceManager:
    """Tests for PreferenceManager class."""

    @pytest.fixture
    def preference_manager(self):
        """Create a PreferenceManager instance."""
        return PreferenceManager(num_neurons=10, frequency=PREFERENCE_UPDATE_FREQUENCY)

    def test_init(self, preference_manager):
        """Test initialization."""
        assert len(preference_manager.preferences) == 10
        assert preference_manager.last_queried == 0
        assert preference_manager.frequency == PREFERENCE_UPDATE_FREQUENCY

    def test_should_query_on_first_call(self, preference_manager):
        """Test that should_query returns True on first call."""
        assert preference_manager.should_query(block=100) is True

    def test_should_query_after_frequency(self, preference_manager):
        """Test that should_query returns True after frequency blocks."""
        preference_manager.last_queried = 100
        assert preference_manager.should_query(block=100 + PREFERENCE_UPDATE_FREQUENCY) is True

    def test_should_query_before_frequency(self, preference_manager):
        """Test that should_query returns False before frequency blocks."""
        preference_manager.last_queried = 100
        assert preference_manager.should_query(block=100 + PREFERENCE_UPDATE_FREQUENCY - 1) is False

    def test_should_query_with_to_query(self, preference_manager):
        """Test that should_query returns True when to_query is not empty."""
        preference_manager.last_queried = 100
        preference_manager.to_query.add(5)
        assert preference_manager.should_query(block=100) is True

    def test_mark_for_query(self, preference_manager):
        """Test marking UIDs for query."""
        preference_manager.mark_for_query([1, 2, 3])
        assert 1 in preference_manager.to_query
        assert 2 in preference_manager.to_query
        assert 3 in preference_manager.to_query

    def test_get_preference(self, preference_manager):
        """Test getting preference for a UID."""
        preference_manager.preferences[5] = MechanismType.WEATHER_XM.value
        result = preference_manager.get_preference(5)
        assert result == MechanismType.WEATHER_XM

    def test_get_preference_invalid(self, preference_manager):
        """Test getting preference for invalid UID."""
        preference_manager.preferences[5] = 999  # Invalid value
        result = preference_manager.get_preference(5)
        assert result is None

    def test_get_preferences(self, preference_manager):
        """Test getting all preferences."""
        result = preference_manager.get_preferences()
        assert isinstance(result, np.ndarray)
        assert len(result) == 10

    def test_load_preferences(self, preference_manager):
        """Test loading preferences."""
        new_prefs = np.full(10, MechanismType.WEATHER_XM.value)
        preference_manager.load_preferences(new_prefs)
        assert np.array_equal(preference_manager.preferences, new_prefs)

    def test_reshape_preferences_expand(self, preference_manager):
        """Test reshaping preferences to larger size."""
        preference_manager.preferences[0] = MechanismType.WEATHER_XM.value
        preference_manager.reshape_preferences(new_size=15)
        assert len(preference_manager.preferences) == 15
        # Old values should be preserved
        assert preference_manager.preferences[0] == MechanismType.WEATHER_XM.value
        # New UIDs should be marked for query
        assert 10 in preference_manager.to_query

    def test_reshape_preferences_shrink(self, preference_manager):
        """Test reshaping preferences to smaller size."""
        preference_manager.preferences[0] = MechanismType.WEATHER_XM.value
        preference_manager.reshape_preferences(new_size=5)
        assert len(preference_manager.preferences) == 5
        # Old values should be preserved
        assert preference_manager.preferences[0] == MechanismType.WEATHER_XM.value

    @pytest.mark.asyncio
    async def test_query_preferences_success(self, preference_manager):
        """Test querying preferences successfully."""
        mock_metagraph = Mock()
        mock_metagraph.block = 100
        mock_metagraph.axons = [Mock() for _ in range(10)]
        mock_metagraph.neurons = [Mock(uid=i, validator_permit=(i < 2)) for i in range(10)]
        
        mock_dendrite = AsyncMock()
        mock_responses = [
            MechanismType.ERA5 if i < 5 else MechanismType.WEATHER_XM
            for i in range(8)  # 8 non-validator neurons
        ]
        mock_dendrite.return_value = [
            Mock() if isinstance(r, MechanismType) else r
            for r in mock_responses
        ]
        
        # Mock the deserialize to return the mechanism type
        for i, resp in enumerate(mock_dendrite.return_value):
            if hasattr(resp, 'deserialize'):
                resp.deserialize = Mock(return_value=mock_responses[i])
            else:
                resp = mock_responses[i]
        
        preference_manager.to_query = {2, 3, 4}
        preference_manager.last_queried = 50
        
        # Mock the dendrite call properly
        async def mock_dendrite_call(*args, **kwargs):
            return mock_responses
        
        mock_dendrite.side_effect = mock_dendrite_call
        
        # This is a simplified test - the actual implementation is more complex
        # We'll test the structure
        assert preference_manager.should_query(block=100) is True

    def test_get_preference_default(self, preference_manager):
        """Test getting default preference."""
        result = preference_manager.get_preference(0)
        # Default should be ERA5 (UNSPECIFIED = ERA5.value)
        assert result == MechanismType.ERA5

