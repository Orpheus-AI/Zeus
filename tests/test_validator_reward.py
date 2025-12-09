"""
Unit tests for zeus.validator.reward module.
"""
import pytest
import torch
import numpy as np
from zeus.validator.reward import (
    help_format_miner_output,
    get_shape_penalty,
    rmse,
    set_penalties,
    get_curved_scores,
    set_rewards,
)
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import (
    REWARD_DIFFICULTY_SCALER,
    REWARD_RMSE_WEIGHT,
    REWARD_EFFICIENCY_WEIGHT,
    MIN_RELATIVE_SCORE,
    MAX_RELATIVE_SCORE,
    CAP_FACTOR_EFFICIENCY,
    EFFICIENCY_THRESHOLD,
)


class TestHelpFormatMinerOutput:
    """Tests for help_format_miner_output function."""

    def test_correct_shape_no_change(self, sample_tensor):
        """Test that correct shape returns unchanged."""
        correct_shape = sample_tensor.shape
        result = help_format_miner_output(correct_shape, sample_tensor)
        assert torch.equal(result, sample_tensor)

    def test_extra_dimension_squeezed(self, sample_tensor):
        """Test that extra dimension is squeezed."""
        correct_shape = sample_tensor.shape
        extra_dim = sample_tensor.unsqueeze(-1)
        result = help_format_miner_output(correct_shape, extra_dim)
        assert result.shape == correct_shape

    def test_wrong_shape_no_squeeze(self):
        """Test that completely wrong shape is not modified."""
        correct_shape = torch.Size([24, 5, 5])
        wrong_tensor = torch.randn(10, 3, 3)
        result = help_format_miner_output(correct_shape, wrong_tensor)
        assert result.shape == wrong_tensor.shape


class TestGetShapePenalty:
    """Tests for get_shape_penalty function."""

    def test_correct_shape_no_penalty(self, sample_tensor):
        """Test that correct shape has no penalty."""
        correct_shape = sample_tensor.shape
        assert get_shape_penalty(correct_shape, sample_tensor) is False

    def test_wrong_shape_penalty(self, sample_tensor):
        """Test that wrong shape has penalty."""
        correct_shape = torch.Size([24, 5, 5])
        wrong_tensor = torch.randn(10, 3, 3)
        assert get_shape_penalty(correct_shape, wrong_tensor) is True

    def test_nan_values_penalty(self, sample_tensor):
        """Test that NaN values result in penalty."""
        correct_shape = sample_tensor.shape
        nan_tensor = sample_tensor.clone()
        nan_tensor[0, 0, 0] = float('nan')
        assert get_shape_penalty(correct_shape, nan_tensor) is True

    def test_inf_values_penalty(self, sample_tensor):
        """Test that Inf values result in penalty."""
        correct_shape = sample_tensor.shape
        inf_tensor = sample_tensor.clone()
        inf_tensor[0, 0, 0] = float('inf')
        assert get_shape_penalty(correct_shape, inf_tensor) is True


class TestRMSE:
    """Tests for rmse function."""

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        output = torch.tensor([1.0, 2.0, 3.0])
        prediction = torch.tensor([1.1, 2.1, 3.1])
        result = rmse(output, prediction)
        expected = np.sqrt(((0.1 ** 2 + 0.1 ** 2 + 0.1 ** 2) / 3))
        assert abs(result - expected) < 1e-6

    def test_rmse_perfect_match(self):
        """Test RMSE with perfect match."""
        output = torch.tensor([1.0, 2.0, 3.0])
        prediction = torch.tensor([1.0, 2.0, 3.0])
        result = rmse(output, prediction)
        assert result == 0.0

    def test_rmse_with_default_on_error(self):
        """Test RMSE returns default on error."""
        output = torch.tensor([1.0, 2.0, 3.0])
        prediction = torch.tensor([1.0, 2.0])  # Wrong shape
        result = rmse(output, prediction, default=999.0)
        assert result == 999.0

    def test_rmse_raises_without_default(self):
        """Test RMSE raises error without default."""
        output = torch.tensor([1.0, 2.0, 3.0])
        prediction = torch.tensor([1.0, 2.0])  # Wrong shape
        with pytest.raises(Exception):
            rmse(output, prediction)


class TestSetPenalties:
    """Tests for set_penalties function."""

    def test_set_penalties_correct_shape(self, sample_tensor):
        """Test penalties with correct shape."""
        correct_shape = sample_tensor.shape
        miners_data = [
            MinerData(
                uid=i,
                hotkey=f"hotkey_{i}",
                response_time=1.0,
                prediction=sample_tensor.clone()
            )
            for i in range(3)
        ]
        result = set_penalties(correct_shape, miners_data)
        assert all(not m.shape_penalty for m in result)

    def test_set_penalties_wrong_shape(self, sample_tensor):
        """Test penalties with wrong shape."""
        correct_shape = torch.Size([24, 5, 5])
        wrong_tensor = torch.randn(10, 3, 3)
        miners_data = [
            MinerData(
                uid=0,
                hotkey="hotkey_0",
                response_time=1.0,
                prediction=wrong_tensor
            )
        ]
        result = set_penalties(correct_shape, miners_data)
        assert result[0].shape_penalty is True
        assert result[0].rmse == -1.0
        assert result[0].score == 0

    def test_set_penalties_mixed(self, sample_tensor):
        """Test penalties with mixed correct and wrong shapes."""
        correct_shape = sample_tensor.shape
        miners_data = [
            MinerData(
                uid=0,
                hotkey="hotkey_0",
                response_time=1.0,
                prediction=sample_tensor.clone()
            ),
            MinerData(
                uid=1,
                hotkey="hotkey_1",
                response_time=1.0,
                prediction=torch.randn(10, 3, 3)
            )
        ]
        result = set_penalties(correct_shape, miners_data)
        assert result[0].shape_penalty is False
        assert result[1].shape_penalty is True


class TestGetCurvedScores:
    """Tests for get_curved_scores function."""

    def test_get_curved_scores_basic(self):
        """Test basic curved scores calculation."""
        raw_scores = [0.1, 0.5, 0.9]
        result = get_curved_scores(raw_scores, gamma=1.0, max_score=1.0)
        assert len(result) == 3
        assert all(0 <= score <= 1.0 for score in result)

    def test_get_curved_scores_gamma_correction(self):
        """Test gamma correction affects scores."""
        raw_scores = [0.1, 0.5, 0.9]
        result_linear = get_curved_scores(raw_scores, gamma=1.0, max_score=1.0)
        result_curved = get_curved_scores(raw_scores, gamma=0.5, max_score=1.0)
        # Curved should have different distribution
        assert result_linear != result_curved

    def test_get_curved_scores_edge_case_equal(self):
        """Test edge case where max_score equals min_score."""
        raw_scores = [0.5, 0.5, 0.5]
        result = get_curved_scores(raw_scores, gamma=1.0, max_score=0.5)
        assert all(score == 1.0 for score in result)

    def test_get_curved_scores_with_cap_factor(self):
        """Test curved scores with cap factor."""
        raw_scores = [0.1, 0.5, 0.9]
        result = get_curved_scores(
            raw_scores, 
            gamma=1.0, 
            max_score=1.0,
            cap_factor=0.5
        )
        assert len(result) == 3
        assert all(0 <= score <= 1.0 for score in result)

    def test_get_curved_scores_with_min_max(self):
        """Test curved scores with explicit min and max."""
        raw_scores = [0.1, 0.5, 0.9]
        result = get_curved_scores(
            raw_scores,
            gamma=1.0,
            max_score=1.0,
            min_score=0.0
        )
        assert len(result) == 3
        assert all(0 <= score <= 1.0 for score in result)


class TestSetRewards:
    """Tests for set_rewards function."""

    def test_set_rewards_basic(self, sample_tensor):
        """Test basic reward calculation."""
        output_data = sample_tensor
        baseline_data = sample_tensor + 0.1  # Slightly worse
        miners_data = [
            MinerData(
                uid=i,
                hotkey=f"hotkey_{i}",
                response_time=0.5 + i * 0.1,
                prediction=sample_tensor.clone() + (i * 0.01)
            )
            for i in range(3)
        ]
        # Remove penalties first
        for m in miners_data:
            m.shape_penalty = False

        difficulty_grid = np.ones((5, 5))
        challenge_age = 0.5

        result = set_rewards(
            output_data=output_data,
            miners_data=miners_data,
            baseline_data=baseline_data,
            difficulty_grid=difficulty_grid,
            challenge_age=challenge_age
        )

        assert len(result) == 3
        for miner in result:
            assert miner.rmse is not None
            assert miner.baseline_improvement is not None
            assert miner.quality_score is not None
            assert miner.efficiency_score is not None
            assert miner.score is not None
            assert 0 <= miner.score <= 1.0

    def test_set_rewards_filters_penalties(self, sample_tensor):
        """Test that penalties are filtered out."""
        output_data = sample_tensor
        baseline_data = sample_tensor
        miners_data = [
            MinerData(
                uid=0,
                hotkey="hotkey_0",
                response_time=1.0,
                prediction=sample_tensor.clone()
            ),
            MinerData(
                uid=1,
                hotkey="hotkey_1",
                response_time=1.0,
                prediction=torch.randn(10, 3, 3)  # Wrong shape
            )
        ]
        miners_data[0].shape_penalty = False
        miners_data[1].shape_penalty = True

        difficulty_grid = np.ones((5, 5))
        challenge_age = 0.5

        result = set_rewards(
            output_data=output_data,
            miners_data=miners_data,
            baseline_data=baseline_data,
            difficulty_grid=difficulty_grid,
            challenge_age=challenge_age
        )

        assert len(result) == 1
        assert result[0].uid == 0

    def test_set_rewards_empty_list(self, sample_tensor):
        """Test with empty miners list."""
        output_data = sample_tensor
        baseline_data = sample_tensor
        miners_data = []
        difficulty_grid = np.ones((5, 5))
        challenge_age = 0.5

        result = set_rewards(
            output_data=output_data,
            miners_data=miners_data,
            baseline_data=baseline_data,
            difficulty_grid=difficulty_grid,
            challenge_age=challenge_age
        )

        assert len(result) == 0

    def test_set_rewards_score_components(self, sample_tensor):
        """Test that score components are properly weighted."""
        output_data = sample_tensor
        baseline_data = sample_tensor + 0.1
        miners_data = [
            MinerData(
                uid=0,
                hotkey="hotkey_0",
                response_time=0.3,  # Fast
                prediction=sample_tensor.clone() + 0.05  # Good prediction
            )
        ]
        miners_data[0].shape_penalty = False

        difficulty_grid = np.ones((5, 5))
        challenge_age = 0.5

        result = set_rewards(
            output_data=output_data,
            miners_data=miners_data,
            baseline_data=baseline_data,
            difficulty_grid=difficulty_grid,
            challenge_age=challenge_age
        )

        miner = result[0]
        expected_score = (
            REWARD_RMSE_WEIGHT * miner.quality_score +
            REWARD_EFFICIENCY_WEIGHT * miner.efficiency_score
        )
        assert abs(miner.score - expected_score) < 1e-6

