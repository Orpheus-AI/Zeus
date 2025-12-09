"""
Unit tests for zeus.base.validator module.
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np
from zeus.base.validator import BaseValidatorNeuron
from zeus.validator.constants import MechanismType
from tests.conftest import create_mock_metagraph, setup_base_validator_mocks


class TestBaseValidatorNeuron:
    """Tests for BaseValidatorNeuron class."""

    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    def test_base_validator_mechagraphs_dict(
        self, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, mock_check_config, 
        mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test that mechagraphs are stored as a dictionary."""
        class ConcreteValidator(BaseValidatorNeuron):
            async def forward(self):
                pass
        
        era5_mg = Mock(mechanism=MechanismType.ERA5, size=128)
        weatherxm_mg = Mock(mechanism=MechanismType.WEATHER_XM, size=128)
        mock_metagraph = create_mock_metagraph(n=1, mechagraphs=[era5_mg, weatherxm_mg])
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        
        assert isinstance(validator.mechagraphs, dict)
        assert validator.mechagraphs[MechanismType.ERA5] == era5_mg
        assert validator.mechagraphs[MechanismType.WEATHER_XM] == weatherxm_mg

    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    def test_base_validator_scores_shape(
        self, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, mock_check_config, 
        mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test that scores have correct shape."""
        class ConcreteValidator(BaseValidatorNeuron):
            async def forward(self):
                pass
        
        mock_metagraph = create_mock_metagraph(n=10)
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        
        assert validator.scores.shape == (10, 2)  # n neurons, 2 mechanisms
        assert validator.scores.dtype == np.float32

    @pytest.mark.asyncio
    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    async def test_base_validator_concurrent_forward(
        self, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, mock_check_config, 
        mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test concurrent_forward method."""
        class ConcreteValidator(BaseValidatorNeuron):
            call_count = 0
            async def forward(self):
                ConcreteValidator.call_count += 1
        
        mock_metagraph = create_mock_metagraph(n=1)
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        validator.config.neuron.num_concurrent_forwards = 3
        
        ConcreteValidator.call_count = 0
        await validator.concurrent_forward()
        
        assert ConcreteValidator.call_count == 3

    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    @patch('zeus.base.validator.get_uids')
    def test_set_weights_calls_set_weights_for_mechanism(
        self, mock_get_uids, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, 
        mock_check_config, mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test that set_weights calls _set_weights_for_mechanism with correct top_mech_scores."""
        class ConcreteValidator(BaseValidatorNeuron):
            async def forward(self):
                pass
        
        era5_mg = Mock(mechanism=MechanismType.ERA5, size=3)
        weatherxm_mg = Mock(mechanism=MechanismType.WEATHER_XM, size=2)
        mock_metagraph = create_mock_metagraph(n=5, mechagraphs=[era5_mg, weatherxm_mg], include_validator_attrs=True)
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        # All UIDs are available (not validators) - get_uids with miners=False returns empty list
        mock_get_uids.return_value = []
        mock_base_validator_config.neuron.vpermit_tao_limit = 0
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        
        # Set up scores: ERA5 scores in column 0, WEATHER_XM in column 1
        # Higher scores for later UIDs
        validator.scores = np.array([
            [0.1, 0.2],  # uid 0
            [0.2, 0.3],  # uid 1
            [0.3, 0.4],  # uid 2
            [0.4, 0.5],  # uid 3
            [0.5, 0.6],  # uid 4
        ], dtype=np.float32)
        
        # Mock preference manager: first 3 prefer ERA5 (0), last 2 prefer WEATHER_XM (1)
        validator.preference_manager.get_preferences = Mock(return_value=np.array([0, 0, 0, 1, 1]))
        
        # Mock _set_weights_for_mechanism to track calls
        validator._set_weights_for_mechanism = Mock()
        
        validator.set_weights()
        
        # Should be called twice (once for each mechanism)
        assert validator._set_weights_for_mechanism.call_count == 2
        
        # Check ERA5 call (size=3, so top 3 miners: uids 2, 3, 4)
        # But after preference mask, only uids 0, 1, 2 have non-zero scores
        # So top 3 from [0.1, 0.2, 0.3, 0, 0] are uids 0, 1, 2
        era5_call = validator._set_weights_for_mechanism.call_args_list[0]
        assert era5_call[0][0] == MechanismType.ERA5
        top_mech_scores_era5 = era5_call[0][1]
        assert top_mech_scores_era5.shape == (5,)
        # Top 3 should have scores (uids 0, 1, 2 after preference filtering)
        assert top_mech_scores_era5[0] == 0.1
        assert top_mech_scores_era5[1] == 0.2
        assert top_mech_scores_era5[2] == 0.3
        assert top_mech_scores_era5[3] == 0.0
        assert top_mech_scores_era5[4] == 0.0
        
        # Check WEATHER_XM call (size=2, so top 2 miners: uids 3, 4)
        # After preference mask, only uids 3, 4 have non-zero scores
        weatherxm_call = validator._set_weights_for_mechanism.call_args_list[1]
        assert weatherxm_call[0][0] == MechanismType.WEATHER_XM
        top_mech_scores_weatherxm = weatherxm_call[0][1]
        assert top_mech_scores_weatherxm.shape == (5,)
        # Top 2 should have scores (uids 3, 4 after preference filtering)
        assert top_mech_scores_weatherxm[3] == 0.5
        assert top_mech_scores_weatherxm[4] == 0.6
        assert top_mech_scores_weatherxm[0] == 0.0
        assert top_mech_scores_weatherxm[1] == 0.0
        assert top_mech_scores_weatherxm[2] == 0.0

    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    def test_update_scores_basic(
        self, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, 
        mock_check_config, mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test update_scores with valid inputs."""
        class ConcreteValidator(BaseValidatorNeuron):
            async def forward(self):
                pass
        
        mock_metagraph = create_mock_metagraph(
            n=5, 
            block_at_registration=np.array([900, 950, 980, 990, 995])  # Different ages
        )
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        
        # Initialize scores
        validator.scores = np.zeros((5, 2), dtype=np.float32)
        
        # Update scores for ERA5 mechanism
        rewards = np.array([0.5, 0.7, 0.9])
        uids = [0, 2, 4]
        
        validator.update_scores(rewards, uids, MechanismType.ERA5)
        
        # Check that scores were updated for the specified UIDs
        assert validator.scores[0, MechanismType.ERA5.value] > 0
        assert validator.scores[2, MechanismType.ERA5.value] > 0
        assert validator.scores[4, MechanismType.ERA5.value] > 0
        # Other UIDs should remain 0
        assert validator.scores[1, MechanismType.ERA5.value] == 0
        assert validator.scores[3, MechanismType.ERA5.value] == 0
        # WEATHER_XM scores should remain unchanged
        assert np.all(validator.scores[:, MechanismType.WEATHER_XM.value] == 0)

    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    def test_update_scores_nan_handling(
        self, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, 
        mock_check_config, mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test that update_scores handles NaN values correctly."""
        class ConcreteValidator(BaseValidatorNeuron):
            async def forward(self):
                pass
        
        mock_metagraph = create_mock_metagraph(n=3)
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        validator.scores = np.zeros((3, 2), dtype=np.float32)
        
        # Test with NaN values
        rewards = np.array([0.5, np.nan, 0.9])
        uids = [0, 1, 2]
        
        validator.update_scores(rewards, uids, MechanismType.ERA5)
        
        # NaN should be replaced with 0
        assert not np.isnan(validator.scores[1, MechanismType.ERA5.value])
        assert validator.scores[1, MechanismType.ERA5.value] == 0

    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    def test_update_scores_empty_arrays(
        self, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, 
        mock_check_config, mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test that update_scores handles empty arrays gracefully."""
        class ConcreteValidator(BaseValidatorNeuron):
            async def forward(self):
                pass
        
        mock_metagraph = create_mock_metagraph(n=3)
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        validator.scores = np.zeros((3, 2), dtype=np.float32)
        initial_scores = validator.scores.copy()
        
        # Test with empty arrays - should return early without error
        validator.update_scores(np.array([]), [], MechanismType.ERA5)
        
        # Scores should remain unchanged
        np.testing.assert_array_equal(validator.scores, initial_scores)

    @patch('zeus.base.neuron.ttl_get_block', return_value=1000)
    @patch('zeus.base.validator.BaseValidatorNeuron.save_state')
    @patch('zeus.base.validator.BaseValidatorNeuron.load_state')
    @patch('zeus.base.neuron.check_config')
    @patch('zeus.base.neuron.bt.wallet')
    @patch('zeus.base.neuron.bt.subtensor')
    @patch('zeus.base.neuron.MetagraphDelegate')
    @patch('zeus.base.validator.ZeusDendrite')
    def test_update_scores_shape_mismatch(
        self, mock_dendrite, mock_delegate, mock_subtensor, mock_wallet, 
        mock_check_config, mock_load_state, mock_save_state, mock_get_block, mock_base_validator_config,
        mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance
    ):
        """Test that update_scores raises ValueError on shape mismatch."""
        class ConcreteValidator(BaseValidatorNeuron):
            async def forward(self):
                pass
        
        mock_metagraph = create_mock_metagraph(n=3)
        setup_base_validator_mocks(
            mock_wallet, mock_subtensor, mock_delegate, mock_dendrite,
            mock_wallet_instance, mock_subtensor_instance, mock_dendrite_instance,
            mock_metagraph
        )
        
        validator = ConcreteValidator(config=mock_base_validator_config)
        validator.scores = np.zeros((3, 2), dtype=np.float32)
        
        # Test with mismatched shapes
        rewards = np.array([0.5, 0.7])  # 2 rewards
        uids = [0, 1, 2]  # 3 UIDs
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            validator.update_scores(rewards, uids, MechanismType.ERA5)

