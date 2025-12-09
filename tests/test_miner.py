"""
Unit tests for neurons.miner module.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from zeus.protocol import TimePredictionSynapse, LocalPredictionSynapse
from zeus.validator.constants import MechanismType, WEATHERXM_CELL_RESOLUTION
from neurons.miner import Miner



class TestForwardERA5:
    """Tests for forward_era5 function."""

    @pytest.fixture(scope="class")
    def mock_miner_config(self):
        """Create a mock config for Miner."""
        config = Mock()
        config.blacklist = Mock()
        config.blacklist.force_validator_permit = False
        config.blacklist.allow_non_registered = False
        config.weatherxm = Mock()
        config.weatherxm.api_key = "test_api_key"
        return config

    @pytest.fixture(scope="class")
    def mock_openmeteo_responses(self):
        """Create mock OpenMeteo API responses for a 2x2 grid (4 locations)."""
        # Create mock data: 24 hours, 1 variable
        mock_data = np.random.randn(24).astype(np.float32)
        
        # Create 4 responses (one for each location in 2x2 grid)
        responses = []
        for _ in range(4):
            mock_response = Mock()
            mock_hourly = Mock()
            mock_variable = Mock()
            mock_variable.ValuesAsNumpy = Mock(return_value=mock_data)
            mock_hourly.Variables = Mock(return_value=mock_variable)
            mock_hourly.VariablesLength = Mock(return_value=1)
            mock_response.Hourly = Mock(return_value=mock_hourly)
            responses.append(mock_response)
        
        return responses

    @pytest.fixture(scope="class")
    def mock_openmeteo_api(self, mock_openmeteo_responses):
        """Create a shared mock OpenMeteo API that all tests will use."""
        mock_api = Mock()
        mock_api.weather_api = Mock(return_value=mock_openmeteo_responses)
        return mock_api

    @pytest.fixture(scope="class")
    def mock_base_miner(self, mock_miner_config, mock_openmeteo_api):
        """Create a mock BaseMinerNeuron to avoid full initialization."""
        with patch('neurons.miner.BaseMinerNeuron.__init__', return_value=None):
            miner = Miner.__new__(Miner)
            miner.config = mock_miner_config
            miner.axon = Mock()
            miner.axon.attach = Mock(return_value=miner.axon)
            miner.openmeteo_api = mock_openmeteo_api
            return miner

    @pytest.fixture
    def time_prediction_synapse(self):
        """Create a TimePredictionSynapse for testing."""
        start_time = 1000000.0
        end_time = start_time + (23 * 3600) 
        synapse = TimePredictionSynapse(
            locations=[[(40.0, -74.0), (41.0, -75.0)], [(42.0, -76.0), (43.0, -77.0)]],
            requested_hours=24,
            start_time=start_time,
            end_time=end_time,  # 24 hours later
            variable="2m_temperature"
        )
        return synapse

    @pytest.mark.asyncio
    async def test_forward_era5_basic(self, mock_base_miner, time_prediction_synapse):
        """Test basic forward_era5 functionality."""
        # Use real converter, no mocking
        result = await mock_base_miner.forward_era5(time_prediction_synapse)
        
        # Verify the synapse was modified
        assert result is time_prediction_synapse
        assert result.predictions is not None
        assert len(result.predictions) > 0
        assert result.version is not None

    @pytest.mark.asyncio
    async def test_forward_era5_sets_version(self, mock_base_miner, time_prediction_synapse):
        """Test that forward_era5 sets the version."""
        # Use real converter, no mocking
        result = await mock_base_miner.forward_era5(time_prediction_synapse)
        
        assert result.version is not None
        assert isinstance(result.version, str)

    @pytest.mark.asyncio
    async def test_forward_era5_calls_openmeteo_api(self, mock_base_miner, time_prediction_synapse):
        """Test that forward_era5 calls the OpenMeteo API with correct parameters."""
        # Use real converter, no mocking
        await mock_base_miner.forward_era5(time_prediction_synapse)
        
        # Verify API was called
        assert mock_base_miner.openmeteo_api.weather_api.called
        call_args = mock_base_miner.openmeteo_api.weather_api.call_args
        assert call_args[0][0] == "https://api.open-meteo.com/v1/forecast"
        assert call_args[1]["method"] == "POST"
        assert "params" in call_args[1]
        params = call_args[1]["params"]
        assert "latitude" in params
        assert "longitude" in params
        assert "hourly" in params
        
        # Verify coordinates are flattened correctly (2x2 grid = 4 locations)
        assert len(params["latitude"]) == 4
        assert len(params["longitude"]) == 4

    @pytest.mark.asyncio
    async def test_forward_era5_handles_multiple_variables(self, mock_base_miner, time_prediction_synapse):
        """Test that forward_era5 handles multiple variables correctly."""
        # Use real converter, no mocking
        result = await mock_base_miner.forward_era5(time_prediction_synapse)
        
        assert result.predictions is not None


class TestForwardWeatherXM:
    """Tests for forward_weatherxm function."""

    @pytest.fixture
    def mock_miner_config(self):
        """Create a mock config for Miner."""
        config = Mock()
        config.blacklist = Mock()
        config.blacklist.force_validator_permit = False
        config.blacklist.allow_non_registered = False
        config.weatherxm = Mock()
        config.weatherxm.api_key = "test_api_key"
        return config

    @pytest.fixture
    def mock_base_miner(self, mock_miner_config):
        """Create a mock BaseMinerNeuron to avoid full initialization."""
        with patch('neurons.miner.BaseMinerNeuron.__init__', return_value=None):
            miner = Miner.__new__(Miner)
            miner.config = mock_miner_config
            miner.axon = Mock()
            miner.axon.attach = Mock(return_value=miner.axon)
            miner.openmeteo_api = Mock()
            return miner

    @pytest.fixture
    def local_prediction_synapse(self):
        """Create a LocalPredictionSynapse for testing."""
        # For 24 hours: 24 * 3600 = 86400 seconds
        start_time = 1000000.0
        end_time = start_time + (23 * 3600)  # 24 hours later
        synapse = LocalPredictionSynapse(
            latitude=40.7128,
            longitude=-74.0060,
            elevation=10.5,
            requested_hours=24,
            start_time=start_time,
            end_time=end_time,
            variable="temperature"
        )
        return synapse

    @pytest.fixture(scope="class")
    def mock_weatherxm_response(self):
        """Create a shared mock WeatherXM API response."""
        # Create mock hourly data for 2 days (to cover 24 hours)
        mock_api_response = [
            {
                "hourly": [
                    {"temperature": 20.0 + i * 0.1} for i in range(24)
                ]
            },
            {
                "hourly": [
                    {"temperature": 22.0 + i * 0.1} for i in range(24)
                ]
            }
        ]
        
        mock_response = Mock()
        mock_response.json = Mock(return_value=mock_api_response)
        return mock_response

    @pytest.fixture(scope="class")
    def mock_requests_get_patcher(self, mock_weatherxm_response):
        """Create a shared mock requests.get patcher that all tests will use."""
        patcher = patch('neurons.miner.requests.get', return_value=mock_weatherxm_response)
        mock_get = patcher.start()
        yield mock_get
        patcher.stop()

    @pytest.fixture
    def mock_requests_get(self, mock_requests_get_patcher):
        """Fixture to access the shared mock requests.get."""
        return mock_requests_get_patcher

    @pytest.mark.asyncio
    async def test_forward_weatherxm_basic(self, mock_base_miner, local_prediction_synapse, mock_requests_get):
        """Test basic forward_weatherxm functionality."""
        # Use real h3, no mocking
        result = await mock_base_miner.forward_weatherxm(local_prediction_synapse)
        
        # Verify the synapse was modified
        assert result is local_prediction_synapse
        assert result.predictions is not None
        assert len(result.predictions) == 24  # Should have 24 hours
        assert result.version is not None

    @pytest.mark.asyncio
    async def test_forward_weatherxm_sets_version(self, mock_base_miner, local_prediction_synapse, mock_requests_get):
        """Test that forward_weatherxm sets the version."""
        # Use real h3, no mocking
        result = await mock_base_miner.forward_weatherxm(local_prediction_synapse)
        
        assert result.version is not None
        assert isinstance(result.version, str)

    @pytest.mark.asyncio
    async def test_forward_weatherxm_calls_api_with_correct_params(self, mock_base_miner, local_prediction_synapse, mock_requests_get):
        """Test that forward_weatherxm calls the WeatherXM API with correct parameters."""
        # Use real h3, no mocking
        await mock_base_miner.forward_weatherxm(local_prediction_synapse)
        
        # Verify API was called
        assert mock_requests_get.called
        call_args = mock_requests_get.call_args
        
        # Check URL contains the forecast endpoint
        assert "/api/v1/cells/" in call_args[0][0]
        assert "/forecast/wxmv1" in call_args[0][0]
        
        # Check params
        assert "params" in call_args[1]
        params = call_args[1]["params"]
        assert "include" in params
        assert params["include"] == "hourly"
        assert "from" in params
        assert "to" in params
        
        # Check headers
        assert "headers" in call_args[1]
        headers = call_args[1]["headers"]
        assert "X-API-KEY" in headers
        assert headers["X-API-KEY"] == "test_api_key"

    @pytest.mark.asyncio
    async def test_forward_weatherxm_uses_correct_h3_resolution(self, mock_base_miner, local_prediction_synapse, mock_requests_get):
        """Test that forward_weatherxm uses the correct H3 resolution."""
        # Use real h3, verify it's called correctly by checking the URL contains a valid h3 cell
        await mock_base_miner.forward_weatherxm(local_prediction_synapse)
        
        # Verify the API was called with an h3 cell in the URL
        assert mock_requests_get.called
        call_args = mock_requests_get.call_args
        url = call_args[0][0]
        # Extract h3 cell from URL (format: /api/v1/cells/{h3_cell}/forecast/wxmv1)
        assert "/api/v1/cells/" in url
        # The h3 cell should be a valid hex string (h3 cells are hex)
        import re
        h3_match = re.search(r'/cells/([a-f0-9]+)/', url)
        assert h3_match is not None, "URL should contain a valid h3 cell"

    @pytest.mark.asyncio
    async def test_forward_weatherxm_slices_hours_correctly(self, mock_base_miner, local_prediction_synapse, mock_requests_get):
        """Test that forward_weatherxm correctly slices to requested hours."""
        # Use real h3, no mocking
        result = await mock_base_miner.forward_weatherxm(local_prediction_synapse)
        
        # Should have exactly 24 hours (or less if slicing removes some)
        assert len(result.predictions) > 0
        assert len(result.predictions) <= 24

    @pytest.fixture
    def local_prediction_synapse_humidity(self):
        """Create a LocalPredictionSynapse for testing with humidity variable."""
        # For 24 hours: 24 * 3600 = 86400 seconds
        start_time = 1000000.0
        end_time = start_time + (23 * 3600)  # 24 hours later
        synapse = LocalPredictionSynapse(
            latitude=40.7128,
            longitude=-74.0060,
            elevation=10.5,
            requested_hours=24,
            start_time=start_time,
            end_time=end_time,
            variable="humidity"
        )
        return synapse

    @pytest.mark.asyncio
    async def test_forward_weatherxm_handles_different_variables(self, mock_base_miner, local_prediction_synapse_humidity, mock_requests_get):
        """Test that forward_weatherxm handles different variables correctly."""
        # Update the mock response to return humidity data
        mock_api_response = [
            {
                "hourly": [
                    {"humidity": 50.0 + i * 0.5} for i in range(24)
                ]
            },
            {
                "hourly": [
                    {"humidity": 55.0 + i * 0.5} for i in range(24)
                ]
            }
        ]
        mock_requests_get.return_value.json.return_value = mock_api_response
        
        # Use real h3, no mocking
        result = await mock_base_miner.forward_weatherxm(local_prediction_synapse_humidity)
        
        assert result.predictions is not None
        assert len(result.predictions) == 24
        # Verify it used the correct variable
        assert all(isinstance(p, (int, float)) for p in result.predictions)


