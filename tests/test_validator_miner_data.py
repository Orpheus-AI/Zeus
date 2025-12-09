"""
Unit tests for zeus.validator.miner_data module.
"""
import pytest
import torch
from zeus.validator.miner_data import MinerData


class TestMinerData:
    """Tests for MinerData dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        miner = MinerData(
            hotkey="test_hotkey",
            response_time=1.5,
            prediction=torch.randn(24, 5, 5)
        )
        assert miner.hotkey == "test_hotkey"
        assert miner.response_time == 1.5
        assert miner.uid is None
        assert miner.score is None

    def test_init_with_uid(self):
        """Test initialization with UID."""
        miner = MinerData(
            uid=5,
            hotkey="test_hotkey",
            response_time=1.5,
            prediction=torch.randn(24, 5, 5)
        )
        assert miner.uid == 5

    def test_shape_penalty_property(self):
        """Test shape_penalty property."""
        miner = MinerData(
            hotkey="test_hotkey",
            response_time=1.5,
            prediction=torch.randn(24, 5, 5)
        )
        assert miner.shape_penalty is None
        
        miner.shape_penalty = True
        assert miner.shape_penalty is True

    def test_shape_penalty_setter_sets_penalty_values(self):
        """Test that setting shape_penalty to True sets penalty values."""
        miner = MinerData(
            hotkey="test_hotkey",
            response_time=1.5,
            prediction=torch.randn(24, 5, 5)
        )
        miner.shape_penalty = True
        assert miner.rmse == -1.0
        assert miner.score == 0

    def test_shape_penalty_setter_does_not_override_existing(self):
        """Test that setting shape_penalty to False doesn't override existing values."""
        miner = MinerData(
            hotkey="test_hotkey",
            response_time=1.5,
            prediction=torch.randn(24, 5, 5)
        )
        miner.rmse = 0.5
        miner.score = 0.8
        miner.shape_penalty = False
        assert miner.rmse == 0.5
        assert miner.score == 0.8

    def test_metrics_property(self):
        """Test metrics property."""
        miner = MinerData(
            uid=0,
            hotkey="test_hotkey",
            response_time=1.5,
            prediction=torch.randn(24, 5, 5)
        )
        miner.rmse = 0.5
        miner.score = 0.8
        miner.quality_score = 0.7
        miner.efficiency_score = 0.6
        miner.shape_penalty = False
        
        metrics = miner.metrics
        assert metrics["RMSE"] == 0.5
        assert metrics["score"] == 0.8
        assert metrics["quality_score"] == 0.7
        assert metrics["efficiency_score"] == 0.6
        assert metrics["shape_penalty"] is False
        assert metrics["response_time"] == 1.5

    def test_metrics_property_with_none_values(self):
        """Test metrics property with None values."""
        miner = MinerData(
            hotkey="test_hotkey",
            response_time=1.5,
            prediction=torch.randn(24, 5, 5)
        )
        metrics = miner.metrics
        assert metrics["RMSE"] is None
        assert metrics["score"] is None
        assert metrics["quality_score"] is None
        assert metrics["efficiency_score"] is None
        assert metrics["shape_penalty"] is None
        assert metrics["response_time"] == 1.5

