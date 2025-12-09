"""
Unit tests for zeus.utils.uids module.
"""
import pytest
import torch
from zeus.utils.uids import is_miner_uid, get_uids


class TestCheckUIDAvailability:
    """Tests for check_uid_availability function."""

    def test_check_uid_availability_serving(self, mock_metagraph):
        """Test that serving UID is available."""
        mock_metagraph.axons[5].is_serving = True
        result = is_miner_uid(mock_metagraph, 5, vpermit_tao_limit=1000)
        assert result is True

    def test_check_uid_availability_not_serving(self, mock_metagraph):
        """Test that non-serving UID is not available."""
        mock_metagraph.axons[5].is_serving = False
        result = is_miner_uid(mock_metagraph, 5, vpermit_tao_limit=1000)
        assert result is False

    def test_check_uid_availability_validator_permit_low_stake(self, mock_metagraph):
        """Test validator with permit but low stake is available."""
        # mock_metagraph.netuid is already set in fixture
        mock_metagraph.validator_permit[5] = True
        mock_metagraph.S[5] = 500  # Below limit
        result = is_miner_uid(mock_metagraph, 5, vpermit_tao_limit=1000)
        assert result is True

    def test_check_uid_availability_validator_permit_high_stake(self, mock_metagraph):
        """Test validator with permit and high stake is not available."""
        # mock_metagraph.netuid is already set in fixture
        mock_metagraph.validator_permit[5] = True
        mock_metagraph.S[5] = 2000  # Above limit
        result = is_miner_uid(mock_metagraph, 5, vpermit_tao_limit=1000)
        assert result is False

    def test_check_uid_availability_no_permit_high_stake(self, mock_metagraph):
        """Test non-validator with high stake is available."""
        # mock_metagraph.netuid is already set in fixture
        mock_metagraph.validator_permit[5] = False
        mock_metagraph.S[5] = 2000  # High stake but no permit
        result = is_miner_uid(mock_metagraph, 5, vpermit_tao_limit=1000)
        assert result is True


class TestGetUIDs:
    """Tests for get_uids function."""

    def test_get_uids_all_serving(self, mock_metagraph):
        """Test getting all miner UIDs when all are serving."""
        # All axons are serving by default in fixture
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=True)
        assert result == list(range(10))
        assert len(result) == 10

    def test_get_uids_some_not_serving(self, mock_metagraph):
        """Test getting miner UIDs excludes non-serving ones."""
        # Make some UIDs not serving
        mock_metagraph.axons[2].is_serving = False
        mock_metagraph.axons[5].is_serving = False
        mock_metagraph.axons[8].is_serving = False
        
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=True)
        assert 2 not in result
        assert 5 not in result
        assert 8 not in result
        assert len(result) == 7
        assert all(uid in result for uid in [0, 1, 3, 4, 6, 7, 9])

    def test_get_uids_excludes_validators_with_high_stake(self, mock_metagraph):
        """Test getting miner UIDs excludes validators with high stake."""
        # Set some UIDs as validators with high stake
        mock_metagraph.validator_permit[1] = True
        mock_metagraph.S[1] = 2000  # Above limit
        mock_metagraph.validator_permit[4] = True
        mock_metagraph.S[4] = 1500  # Above limit
        
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=True)
        assert 1 not in result
        assert 4 not in result
        assert len(result) == 8

    def test_get_uids_includes_validators_with_low_stake(self, mock_metagraph):
        """Test getting miner UIDs includes validators with low stake."""
        # Set some UIDs as validators with low stake
        mock_metagraph.validator_permit[2] = True
        mock_metagraph.S[2] = 500  # Below limit
        mock_metagraph.validator_permit[7] = True
        mock_metagraph.S[7] = 800  # Below limit
        
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=True)
        assert 2 in result
        assert 7 in result
        assert len(result) == 10

    def test_get_uids_combined_conditions(self, mock_metagraph):
        """Test getting miner UIDs with multiple exclusion conditions."""
        # UID 0: serving, no permit, low stake -> included
        # UID 1: not serving -> excluded
        # UID 2: serving, validator permit, high stake -> excluded
        # UID 3: serving, validator permit, low stake -> included
        # UID 4: serving, no permit, high stake -> included (no permit)
        
        mock_metagraph.axons[1].is_serving = False
        mock_metagraph.validator_permit[2] = True
        mock_metagraph.S[2] = 2000
        mock_metagraph.validator_permit[3] = True
        mock_metagraph.S[3] = 500
        mock_metagraph.validator_permit[4] = False
        mock_metagraph.S[4] = 2000
        
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=True)
        assert 0 in result
        assert 1 not in result
        assert 2 not in result
        assert 3 in result
        assert 4 in result
        assert len(result) == 8

    def test_get_uids_non_miners(self, mock_metagraph):
        """Test getting non-miner UIDs (miners=False)."""
        # Make some UIDs not serving (non-miners)
        mock_metagraph.axons[2].is_serving = False
        mock_metagraph.axons[5].is_serving = False
        
        # Make some validators with high stake (non-miners)
        mock_metagraph.validator_permit[1] = True
        mock_metagraph.S[1] = 2000
        
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=False)
        assert 1 in result  # Validator with high stake
        assert 2 in result  # Not serving
        assert 5 in result  # Not serving
        assert 0 not in result  # Serving miner
        assert 3 not in result  # Serving miner
        assert len(result) == 3

    def test_get_uids_empty_metagraph(self, mock_metagraph):
        """Test getting UIDs from empty metagraph."""
        mock_metagraph.n = torch.tensor(0)
        mock_metagraph.axons = []
        mock_metagraph.validator_permit = torch.zeros(0, dtype=torch.bool)
        mock_metagraph.S = torch.zeros(0)
        
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=True)
        assert result == []

    def test_get_uids_all_excluded(self, mock_metagraph):
        """Test getting miner UIDs when all are excluded."""
        # Make all UIDs not serving
        for axon in mock_metagraph.axons:
            axon.is_serving = False
        
        result = get_uids(mock_metagraph, vpermit_tao_limit=1000, miners=True)
        assert result == []

