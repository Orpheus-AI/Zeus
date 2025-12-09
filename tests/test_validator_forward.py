"""
Unit tests for zeus.validator.forward module.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from zeus.validator.forward import (
    sample_mechanism,
    parse_miner_inputs,
    complete_challenge,
    do_wandb_logging,
)
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import MechanismType
# MockSample is defined in conftest.py - import using relative import
from .conftest import MockSample


class TestSampleMechanism:
    """Tests for sample_mechanism function."""

    def test_sample_mechanism_returns_mechanism(self, mock_validator):
        """Test that sample_mechanism returns a mechanism."""
        result = sample_mechanism(mock_validator)
        assert result in [MechanismType.ERA5, MechanismType.WEATHER_XM]

    def test_sample_mechanism_only_ready(self, mock_validator):
        """Test that sample_mechanism only samples from ready loaders."""
        mock_validator.data_loaders[MechanismType.WEATHER_XM].is_ready = Mock(return_value=False)
        result = sample_mechanism(mock_validator)
        assert result == MechanismType.ERA5


class TestParseMinerInputs:
    """Tests for parse_miner_inputs function."""

    @pytest.fixture
    def mock_validator(self, mock_metagraph, mock_validator_config):
        """Create a mock validator."""
        validator = Mock()
        validator.metagraph = mock_metagraph
        validator.config = mock_validator_config
        return validator

    def test_parse_miner_inputs_basic(self, mock_validator, sample_tensor):
        """Test basic parsing of miner inputs."""
        sample = MockSample(baseline=sample_tensor)
        hotkeys = ["hotkey_0", "hotkey_1"]
        predictions = [sample_tensor.clone(), sample_tensor.clone()]
        response_times = [1.0, 2.0]
        
        result = parse_miner_inputs(
            mock_validator,
            sample=sample,
            hotkeys=hotkeys,
            predictions=predictions,
            response_times=response_times
        )
        
        assert len(result) == 2
        assert all(isinstance(m, MinerData) for m in result)
        assert result[0].uid == 0
        assert result[1].uid == 1

    def test_parse_miner_inputs_unknown_hotkey(self, mock_validator, sample_tensor):
        """Test parsing with unknown hotkey."""
        sample = MockSample(baseline=sample_tensor)
        hotkeys = ["unknown_hotkey"]
        predictions = [sample_tensor.clone()]
        response_times = [1.0]
        
        result = parse_miner_inputs(
            mock_validator,
            sample=sample,
            hotkeys=hotkeys,
            predictions=predictions,
            response_times=response_times
        )
        
        # Unknown hotkey should not create MinerData
        assert len(result) == 0

    def test_parse_miner_inputs_sets_penalties(self, mock_validator, sample_tensor):
        """Test that penalties are set during parsing."""
        sample = MockSample(baseline=sample_tensor)
        hotkeys = ["hotkey_0"]
        predictions = [torch.randn(10, 3, 3)]  # Wrong shape
        response_times = [1.0]
        
        result = parse_miner_inputs(
            mock_validator,
            sample=sample,
            hotkeys=hotkeys,
            predictions=predictions,
            response_times=response_times
        )
        
        assert len(result) == 1
        assert result[0].shape_penalty is True

    def test_parse_miner_inputs_none_response_time(self, mock_validator, sample_tensor):
        """Test parsing with None response time."""
        sample = MockSample(baseline=sample_tensor)
        hotkeys = ["hotkey_0"]
        predictions = [sample_tensor.clone()]
        response_times = [None]
        
        result = parse_miner_inputs(
            mock_validator,
            sample=sample,
            hotkeys=hotkeys,
            predictions=predictions,
            response_times=response_times
        )
        
        assert len(result) == 1
        assert result[0].response_time == mock_validator.config.neuron.timeout


class TestCompleteChallenge:
    """Tests for complete_challenge function."""

    def test_complete_challenge_basic(self, mock_validator, sample_tensor):
        """Test basic challenge completion."""
        sample = MockSample(
            baseline=sample_tensor,
            output_data=sample_tensor.clone()
        )
        hotkeys = ["hotkey_0", "hotkey_1"]
        predictions = [sample_tensor.clone(), sample_tensor.clone()]
        response_times = [1.0, 2.0]
        
        complete_challenge(
            mock_validator,
            sample=sample,
            hotkeys=hotkeys,
            predictions=predictions,
            response_times=response_times
        )
        
        # Should call update_scores
        assert mock_validator.update_scores.called

    def test_complete_challenge_filters_penalties(self, mock_validator, sample_tensor):
        """Test that penalties are filtered out."""
        sample = MockSample(
            baseline=sample_tensor,
            output_data=sample_tensor.clone()
        )
        hotkeys = ["hotkey_0", "hotkey_1"]
        predictions = [
            sample_tensor.clone(),  # Good
            torch.randn(10, 3, 3)   # Bad shape
        ]
        response_times = [1.0, 2.0]
        
        complete_challenge(
            mock_validator,
            sample=sample,
            hotkeys=hotkeys,
            predictions=predictions,
            response_times=response_times
        )
        
        # Should still call update_scores (with filtered miners)
        assert mock_validator.update_scores.called


class TestDoWandbLogging:
    """Tests for do_wandb_logging function."""

    @patch('zeus.validator.forward.wandb')
    def test_do_wandb_logging_wandb_off(self, mock_wandb):
        """Test that logging is skipped when wandb is off."""
        validator = Mock()
        validator.config = Mock()
        validator.config.wandb = Mock()
        validator.config.wandb.off = True
        
        challenge = MockSample()
        miners_data = [MinerData(
            uid=0,
            hotkey="hotkey_0",
            response_time=1.0,
            prediction=torch.randn(24, 5, 5)
        )]
        
        do_wandb_logging(validator, challenge, miners_data)
        mock_wandb.log.assert_not_called()

    @patch('zeus.validator.forward.wandb')
    def test_do_wandb_logging_logs_metrics(self, mock_wandb):
        """Test that metrics are logged."""
        validator = Mock()
        validator.config = Mock()
        validator.config.wandb = Mock()
        validator.config.wandb.off = False
        
        challenge = MockSample(output_data=torch.randn(24, 5, 5))
        miners_data = [MinerData(
            uid=0,
            hotkey="hotkey_0",
            response_time=1.0,
            prediction=torch.randn(24, 5, 5)
        )]
        miners_data[0].rmse = 0.5
        miners_data[0].score = 0.8
        
        do_wandb_logging(validator, challenge, miners_data)
        assert mock_wandb.log.called

