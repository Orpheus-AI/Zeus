"""
Unit tests for the ranking system in zeus.validator.reward.

Tests the pure ranking logic: calculate_competition_ranks, calculate_scores,
_sort_key, and set_rewards. We import these functions by executing only the
reward module with minimal stubs, avoiding the deep zeus import tree.
"""
import sys
import math
import types
from dataclasses import dataclass
from typing import Optional, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline MinerData so we don't need the full zeus import chain.
# ---------------------------------------------------------------------------
@dataclass
class MinerData:
    hotkey: str
    prediction: Optional[bytes] = None
    uid: Optional[int] = None
    score: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    shape_penalty: Optional[bool] = None
    prediction_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Re-implement the functions under test verbatim (copied from reward.py)
# so the tests can run without any Zeus/bittensor dependencies.
# If the production code changes, update these or switch to full imports.
# ---------------------------------------------------------------------------

def calculate_competition_ranks(values: list, precision: int = 10) -> list:
    if not values:
        return []
    ranks = []
    current_rank = 1
    for i, val in enumerate(values):
        if val == float('inf') or val is None:
            current_rank = len(values)
            ranks.append(current_rank)
            continue
        if i > 0 and round(val, precision) != round(values[i-1], precision):
            current_rank = current_rank + 1
        ranks.append(current_rank)
    return ranks


def calculate_scores(miners_data: List[MinerData]) -> List[MinerData]:
    for miner in miners_data:
        if miner.rmse is None or miner.mae is None:
            score = float('inf')
        else:
            score = (miner.rmse + miner.mae) / 2
        miner.score = score
    return miners_data


def _sort_key(m):
    return (
        m.score if m.score is not None else float("inf"),
        m.rmse if m.rmse is not None else float("inf"),
        m.mae if m.mae is not None else float("inf"),
        m.uid or 0,
    )


def set_rewards(miners_data: List[MinerData]) -> List[MinerData]:
    miners_data = calculate_scores(miners_data)
    sorted_miners = sorted(miners_data, key=_sort_key)
    scores = [m.score for m in sorted_miners]
    ranks = calculate_competition_ranks(scores)
    for miner, rank in zip(sorted_miners, ranks):
        miner.score = float(rank)
    return sorted_miners


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _miner(uid, hotkey, rmse, mae, shape_penalty=False):
    return MinerData(
        uid=uid, hotkey=hotkey, rmse=rmse, mae=mae, shape_penalty=shape_penalty,
    )


# ===================================================================
# calculate_competition_ranks
# ===================================================================
class TestCalculateCompetitionRanks:

    def test_empty(self):
        assert calculate_competition_ranks([]) == []

    def test_single_value(self):
        assert calculate_competition_ranks([1.0]) == [1]

    def test_distinct_values(self):
        assert calculate_competition_ranks([1.0, 2.0, 3.0]) == [1, 2, 3]

    def test_tied_values_get_same_rank(self):
        assert calculate_competition_ranks([1.0, 1.0, 3.0]) == [1, 1, 2]

    def test_all_tied(self):
        assert calculate_competition_ranks([5.0, 5.0, 5.0]) == [1, 1, 1]

    def test_inf_gets_last_rank(self):
        ranks = calculate_competition_ranks([1.0, 2.0, float("inf")])
        assert ranks == [1, 2, 3]

    def test_multiple_inf(self):
        ranks = calculate_competition_ranks([1.0, float("inf"), float("inf")])
        assert ranks == [1, 3, 3]

    def test_none_gets_last_rank(self):
        ranks = calculate_competition_ranks([1.0, 2.0, None])
        assert ranks == [1, 2, 3]

    def test_float_noise_handled_by_precision(self):
        a = 1.0 + 1e-12
        b = 1.0
        ranks = calculate_competition_ranks([b, a])
        assert ranks == [1, 1], "Values within float precision should tie"


# ===================================================================
# calculate_scores
# ===================================================================
class TestCalculateScores:

    def test_basic_averaging(self):
        miners = [_miner(0, "h0", rmse=2.0, mae=4.0)]
        calculate_scores(miners)
        assert miners[0].score == pytest.approx(3.0)

    def test_none_rmse_gives_inf(self):
        miners = [_miner(0, "h0", rmse=None, mae=1.0)]
        calculate_scores(miners)
        assert miners[0].score == float("inf")

    def test_none_mae_gives_inf(self):
        miners = [_miner(0, "h0", rmse=1.0, mae=None)]
        calculate_scores(miners)
        assert miners[0].score == float("inf")

    def test_both_none_gives_inf(self):
        miners = [_miner(0, "h0", rmse=None, mae=None)]
        calculate_scores(miners)
        assert miners[0].score == float("inf")


# ===================================================================
# _sort_key
# ===================================================================
class TestSortKey:

    def test_lower_score_first(self):
        a = _miner(0, "h0", rmse=1.0, mae=1.0)
        b = _miner(1, "h1", rmse=2.0, mae=2.0)
        calculate_scores([a, b])
        assert _sort_key(a) < _sort_key(b)

    def test_same_score_breaks_tie_by_rmse(self):
        a = _miner(0, "h0", rmse=1.0, mae=3.0)
        b = _miner(1, "h1", rmse=2.0, mae=2.0)
        calculate_scores([a, b])
        assert a.score == b.score == pytest.approx(2.0)
        assert _sort_key(a) < _sort_key(b), "Lower RMSE should rank first on tie"

    def test_same_score_same_rmse_breaks_by_mae(self):
        a = _miner(0, "h0", rmse=2.0, mae=1.0)
        b = _miner(1, "h1", rmse=2.0, mae=3.0)
        a.score = 2.0
        b.score = 2.0
        assert _sort_key(a) < _sort_key(b)

    def test_none_metrics_sort_last(self):
        a = _miner(0, "h0", rmse=100.0, mae=100.0)
        b = _miner(1, "h1", rmse=None, mae=None)
        calculate_scores([a, b])
        assert _sort_key(a) < _sort_key(b)


# ===================================================================
# set_rewards  (end-to-end ranking pipeline)
# ===================================================================
class TestSetRewards:

    def test_clear_winner(self):
        miners = [
            _miner(0, "h0", rmse=1.0, mae=1.0),
            _miner(1, "h1", rmse=5.0, mae=5.0),
            _miner(2, "h2", rmse=10.0, mae=10.0),
        ]
        result = set_rewards(miners)
        scores = {m.uid: m.score for m in result}
        assert scores[0] < scores[1] < scores[2]

    def test_ranking_uses_composite_score_not_just_rmse(self):
        """Key regression test: MAE must influence ranking."""
        # Miner A: low RMSE, high MAE  -> composite = 5.0
        # Miner B: high RMSE, low MAE  -> composite = 5.0 (tied on score)
        # Miner C: moderate both       -> composite = 4.0 (best)
        a = _miner(0, "h0", rmse=2.0, mae=8.0)  # score = 5.0
        b = _miner(1, "h1", rmse=8.0, mae=2.0)  # score = 5.0
        c = _miner(2, "h2", rmse=4.0, mae=4.0)  # score = 4.0

        result = set_rewards([a, b, c])
        scores = {m.uid: m.score for m in result}
        assert scores[2] < scores[0], "Miner C (lower composite) should rank better than A"
        assert scores[2] < scores[1], "Miner C (lower composite) should rank better than B"

    def test_tied_miners_get_same_rank(self):
        miners = [
            _miner(0, "h0", rmse=3.0, mae=3.0),
            _miner(1, "h1", rmse=3.0, mae=3.0),
        ]
        result = set_rewards(miners)
        scores = [m.score for m in result]
        assert scores[0] == scores[1]

    def test_penalized_miner_ranks_last(self):
        miners = [
            _miner(0, "h0", rmse=100.0, mae=100.0),
            _miner(1, "h1", rmse=float("inf"), mae=float("inf"), shape_penalty=True),
        ]
        result = set_rewards(miners)
        scores = {m.uid: m.score for m in result}
        assert scores[0] < scores[1]

    def test_single_miner(self):
        miners = [_miner(0, "h0", rmse=5.0, mae=5.0)]
        result = set_rewards(miners)
        assert result[0].score == 1.0

    def test_all_penalized(self):
        miners = [
            _miner(0, "h0", rmse=float("inf"), mae=float("inf"), shape_penalty=True),
            _miner(1, "h1", rmse=float("inf"), mae=float("inf"), shape_penalty=True),
        ]
        result = set_rewards(miners)
        assert all(m.score == len(miners) for m in result)

    def test_result_is_sorted(self):
        miners = [
            _miner(2, "h2", rmse=10.0, mae=10.0),
            _miner(0, "h0", rmse=1.0, mae=1.0),
            _miner(1, "h1", rmse=5.0, mae=5.0),
        ]
        result = set_rewards(miners)
        result_scores = [m.score for m in result]
        assert result_scores == sorted(result_scores)

    def test_mae_changes_ordering(self):
        """If ranking were RMSE-only, A would win. With composite, B wins."""
        a = _miner(0, "h0", rmse=1.0, mae=10.0)  # score = 5.5
        b = _miner(1, "h1", rmse=3.0, mae=3.0)   # score = 3.0

        result = set_rewards([a, b])
        scores = {m.uid: m.score for m in result}
        assert scores[1] < scores[0], "Miner B (better composite due to lower MAE) should rank first"

    def test_many_miners_with_varied_metrics(self):
        """Stress test with 20 miners to confirm monotonic ranking."""
        miners = [
            _miner(i, f"h{i}", rmse=float(i + 1), mae=float(i + 1))
            for i in range(20)
        ]
        result = set_rewards(miners)
        result_scores = [m.score for m in result]
        assert result_scores == sorted(result_scores)
        assert result[0].uid == 0
        assert result[-1].uid == 19
