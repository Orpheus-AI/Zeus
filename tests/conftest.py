"""
Pytest configuration and shared fixtures for Zeus-V2 tests.
"""
import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import bittensor as bt
from zeus.validator.constants import MechanismType
from zeus.validator.miner_data import MinerData
from zeus.data.base.sample import BaseSample

# Register custom pytest marks
pytest_plugins = []

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "network: marks tests that require network access (deselect with '-m \"not network\"')")
    
    # Suppress warnings from bittensor library
    config.addinivalue_line(
        "filterwarnings",
        "ignore::pydantic.warnings.PydanticDeprecatedSince211"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:coroutine.*was never awaited:RuntimeWarning"
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(24, 5, 5)  # 24 hours, 5x5 grid


@pytest.fixture
def sample_tensor_2d():
    """Create a 2D sample tensor for testing."""
    return torch.randn(5, 5)


@pytest.fixture
def sample_tensor_1d():
    """Create a 1D sample tensor for testing."""
    return torch.randn(24)


@pytest.fixture
def mock_metagraph():
    """Create a mock metagraph for testing."""
    metagraph = Mock(spec=bt.Metagraph)
    metagraph.n = torch.tensor(10)
    metagraph.netuid = 18  # Mainnet UID, matches config.netuid
    metagraph.axons = [Mock() for _ in range(10)]
    metagraph.hotkeys = [f"hotkey_{i}" for i in range(10)]
    metagraph.uids = torch.arange(10)
    metagraph.block = torch.tensor(1000)
    metagraph.block_at_registration = torch.zeros(10)
    metagraph.validator_permit = torch.zeros(10, dtype=torch.bool)
    metagraph.S = torch.zeros(10)  # Stake
    metagraph.last_update = torch.zeros(10)
    metagraph.hparams = Mock()
    metagraph.hparams.immunity_period = 100
    metagraph.max_uids = 128
    metagraph.neurons = [Mock(uid=i, hotkey=f"hotkey_{i}") for i in range(10)]
    
    for i, axon in enumerate(metagraph.axons):
        axon.hotkey = f"hotkey_{i}"
        axon.is_serving = True
    
    return metagraph


@pytest.fixture
def mock_validator_config():
    """Create a mock validator config."""
    config = Mock()
    config.neuron = Mock()
    config.neuron.sample_size = 5
    config.neuron.timeout = 12.0
    config.neuron.vpermit_tao_limit = 1000
    config.neuron.moving_average_alpha_min = 0.1
    config.neuron.moving_average_alpha_max = 0.9
    config.neuron.epoch_length = 100
    config.neuron.disable_set_weights = False
    config.neuron.num_concurrent_forwards = 1
    config.neuron.axon_off = False
    config.neuron.full_path = tempfile.mkdtemp()
    config.netuid = 18
    config.subtensor = Mock()
    config.subtensor.chain_endpoint = "mock_endpoint"
    config.logging = Mock()
    config.wandb = Mock()
    config.wandb.off = True
    config.mock = True
    return config


@pytest.fixture
def mock_base_validator_config():
    """Create a mock config for BaseValidatorNeuron tests."""
    config = MagicMock()
    # Make config dict-like for merge operation
    config.items.return_value = {}
    config.neuron = Mock()
    config.neuron.device = "cpu"
    config.neuron.epoch_length = 100
    config.neuron.disable_set_weights = False
    config.neuron.num_concurrent_forwards = 1
    config.neuron.axon_off = True
    config.neuron.full_path = "/tmp/test_validator"
    config.neuron.moving_average_alpha_min = 0.01
    config.neuron.moving_average_alpha_max = 0.1
    config.neuron.name = "test_validator"
    config.netuid = 18
    config.subtensor = Mock()
    config.subtensor.chain_endpoint = "mock_endpoint"
    config.logging = Mock()
    config.logging.logging_dir = "~/.bittensor/miners"
    config.wallet = Mock()
    config.wallet.name = "test_wallet"
    config.wallet.hotkey = "test_hotkey"
    config.wandb = Mock()
    config.wandb.off = True
    config.mock = False  # Set to False so BaseNeuron creates subtensor and metagraph
    return config


@pytest.fixture
def mock_wallet_instance():
    """Create a mock wallet instance for BaseValidatorNeuron tests."""
    mock_wallet = Mock()
    mock_wallet.hotkey = Mock()
    mock_wallet.hotkey.ss58_address = "test_hotkey"
    return mock_wallet


@pytest.fixture
def mock_subtensor_instance():
    """Create a mock subtensor instance for BaseValidatorNeuron tests."""
    mock_subtensor = Mock()
    mock_subtensor.is_hotkey_registered = Mock(return_value=True)
    return mock_subtensor


@pytest.fixture
def mock_dendrite_instance():
    """Create a mock dendrite instance for BaseValidatorNeuron tests."""
    return Mock()


def create_mock_metagraph(n=1, mechagraphs=None, block=1000, block_at_registration=None, 
                          include_validator_attrs=False):
    """Helper function to create a mock metagraph with specified configuration."""
    if mechagraphs is None:
        era5_mg = Mock(mechanism=MechanismType.ERA5, size=128)
        weatherxm_mg = Mock(mechanism=MechanismType.WEATHER_XM, size=128)
        mechagraphs = [era5_mg, weatherxm_mg]
    
    if block_at_registration is None:
        block_at_registration = [900] * n
    
    mock_metagraph = Mock()
    mock_metagraph.hotkeys = ["test_hotkey"] * n
    mock_metagraph.last_update = [0] * n
    mock_metagraph.n = n
    mock_metagraph.max_uids = 256
    mock_metagraph.mechagraphs = mechagraphs
    mock_metagraph.block = block
    mock_metagraph.block_at_registration = np.array(block_at_registration) if isinstance(block_at_registration, list) else block_at_registration
    mock_metagraph.hparams = Mock()
    mock_metagraph.hparams.immunity_period = 100
    mock_metagraph.uids = np.array(list(range(n)))
    
    # Set up axons
    mock_axons = []
    for i in range(n):
        mock_axon = Mock()
        mock_axon.is_serving = True
        mock_axons.append(mock_axon)
    mock_metagraph.axons = mock_axons
    
    # Add validator-specific attributes if needed
    if include_validator_attrs:
        mock_metagraph.validator_permit = np.array([False] * n)
        mock_metagraph.S = np.array([0.0] * n)
    
    # Set up neurons
    mock_neurons = []
    for i in range(n):
        mock_neuron = Mock()
        mock_neuron.uid = i
        mock_neuron.hotkey = "test_hotkey"
        mock_neurons.append(mock_neuron)
    mock_metagraph.neurons = mock_neurons
    mock_metagraph.sync = Mock()
    
    return mock_metagraph


def setup_base_validator_mocks(mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
                                mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
                                mock_metagraph):
    """Helper function to set up common mocks for BaseValidatorNeuron tests."""
    mock_wallet.return_value = mock_wallet_instance
    mock_subtensor.return_value = mock_subtensor_instance
    mock_delegate.return_value = mock_metagraph
    mock_dendrite.return_value = mock_dendrite_instance


@pytest.fixture
def mock_wallet():
    """Create a mock wallet."""
    wallet = Mock(spec=bt.Wallet)
    wallet.hotkey = Mock()
    wallet.hotkey.ss58_address = "test_hotkey_address"
    return wallet


@pytest.fixture
def mock_dendrite():
    """Create a mock dendrite."""
    dendrite = Mock()
    return dendrite


@pytest.fixture
def sample_miner_data():
    """Create sample MinerData for testing."""
    return MinerData(
        uid=0,
        hotkey="test_hotkey",
        response_time=1.5,
        prediction=torch.randn(24, 5, 5)
    )


@pytest.fixture
def sample_miner_data_list():
    """Create a list of sample MinerData for testing."""
    return [
        MinerData(
            uid=i,
            hotkey=f"hotkey_{i}",
            response_time=1.0 + i * 0.1,
            prediction=torch.randn(24, 5, 5)
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader."""
    loader = Mock()
    loader.mechanism = MechanismType.ERA5
    loader.is_ready = Mock(return_value=True)
    loader.get_sample = Mock()
    loader.get_output = Mock(return_value=torch.randn(24, 5, 5))
    loader.get_last_available = Mock(return_value=Mock(timestamp=Mock(return_value=1000000.0)))
    loader.get_relative_age = Mock(return_value=0.5)
    loader.sample_cls = Mock()
    return loader


@pytest.fixture
def mock_baseline_loader():
    """Create a mock baseline loader."""
    loader = Mock()
    loader.get_forecast = Mock(return_value=torch.randn(24, 5, 5))
    return loader


@pytest.fixture
def mock_difficulty_loader():
    """Create a mock difficulty loader."""
    loader = Mock()
    loader.get_difficulty_grid = Mock(return_value=np.ones((5, 5)))
    return loader


@pytest.fixture
def mock_preference_manager():
    """Create a mock preference manager."""
    from zeus.validator.preference import PreferenceManager
    manager = Mock(spec=PreferenceManager)
    manager.get_preference = Mock(return_value=MechanismType.ERA5)
    manager.get_preferences = Mock(return_value=np.full(10, MechanismType.ERA5.value))
    manager.should_query = Mock(return_value=False)
    manager.query_preferences = Mock(return_value=([], []))
    manager.mark_for_query = Mock()
    manager.load_preferences = Mock()
    manager.reshape_preferences = Mock()
    return manager


@pytest.fixture
def mock_validator(mock_metagraph, mock_validator_config):
    """Create a mock validator for testing."""
    validator = Mock()
    validator.metagraph = mock_metagraph
    validator.config = mock_validator_config
    validator.data_loaders = {
        MechanismType.ERA5: Mock(
            is_ready=Mock(return_value=True),
            get_relative_age=Mock(return_value=0.5)
        ),
        MechanismType.WEATHER_XM: Mock(
            is_ready=Mock(return_value=True),
            get_relative_age=Mock(return_value=0.5)
        )
    }
    validator.difficulty_loader = Mock()
    validator.difficulty_loader.get_difficulty_grid = Mock(return_value=np.ones((5, 5)))
    validator.update_scores = Mock()
    return validator


# MockSample class - available to all test files via conftest auto-import
class MockSample(BaseSample):
    """Mock sample for testing. Shared across test files."""
    
    def __init__(self, **kwargs):
        # BaseSample requires either output_data or hours_to_predict
        # If neither is provided, calculate hours_to_predict from timestamps
        start_ts = kwargs.get('start_timestamp', 1000000.0)
        end_ts = kwargs.get('end_timestamp', 1000024.0)
        
        if 'output_data' not in kwargs and 'hours_to_predict' not in kwargs:
            # Calculate hours between start and end timestamps (inclusive)
            # Convert seconds to hours and round to integer
            kwargs['hours_to_predict'] = int((end_ts - start_ts) / 3600) + 1
        
        # Extract MockSample-specific attributes before passing to BaseSample
        # These are not accepted by BaseSample.__init__()
        lat_start = kwargs.pop('lat_start', -10.0)
        lat_end = kwargs.pop('lat_end', 10.0)
        lon_start = kwargs.pop('lon_start', -10.0)
        lon_end = kwargs.pop('lon_end', 10.0)
        ifs_hres_baseline = kwargs.pop('ifs_hres_baseline', None)
        
        # baseline is accepted by BaseSample, but we want to set a default if not provided
        if 'baseline' not in kwargs:
            kwargs['baseline'] = torch.randn(24, 5, 5)
        
        # Get variable before filtering
        variable = kwargs.get('variable', '2m_temperature')
        
        # Filter out attributes that BaseSample doesn't accept
        # BaseSample accepts: start_timestamp, end_timestamp, variable, inserted_at, 
        #                     output_data, hours_to_predict, baseline
        # Note: variable, start_timestamp, and end_timestamp are passed explicitly, not in filtered_kwargs
        base_sample_keys = {'inserted_at', 'output_data', 'hours_to_predict', 'baseline'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_sample_keys}
        
        super().__init__(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            variable=variable,
            **filtered_kwargs
        )
        
        # Set MockSample-specific attributes (not part of BaseSample)
        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lon_start = lon_start
        self.lon_end = lon_end
        self.ifs_hres_baseline = ifs_hres_baseline
    
    def get_synapse(self):
        pass
    
    def get_bbox(self):
        return (self.lat_start, self.lat_end, self.lon_start, self.lon_end)
    
    @property
    def mechanism(self):
        return MechanismType.ERA5


@pytest.fixture
def mock_sample_class():
    """Fixture that provides the MockSample class for use in tests."""
    return MockSample

