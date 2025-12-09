"""
Unit tests for zeus.validator.database module.
"""
import pytest
import torch
import numpy as np
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch
from zeus.validator.database import ResponseDatabase, serialize, deserialize, get_column_names
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import MechanismType
# MockSample is defined in conftest.py - import using relative import
from .conftest import MockSample


class TestResponseDatabase:
    """Tests for ResponseDatabase class."""

    def test_init_creates_tables(self, temp_db_path):
        """Test that initialization creates tables."""
        db = ResponseDatabase(db_path=temp_db_path)
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Check that tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert 'challenges' in tables
        assert 'challenges_weatherxm' in tables
        assert 'responses' in tables
        
        conn.close()

    def test_should_score_returns_false_when_not_ready(self, temp_db_path):
        """Test should_score returns False when dataloader not ready."""
        db = ResponseDatabase(db_path=temp_db_path)
        loader = Mock()
        loader.is_ready = Mock(return_value=False)
        loader.mechanism = MechanismType.ERA5
        
        assert db.should_score(block=1000, dataloader=loader) is False

    def test_should_score_returns_true_after_threshold(self, temp_db_path):
        """Test should_score returns True after block threshold."""
        db = ResponseDatabase(db_path=temp_db_path)
        loader = Mock()
        loader.is_ready = Mock(return_value=True)
        loader.mechanism = MechanismType.ERA5
        
        # First call should return False (not enough blocks)
        assert db.should_score(block=100, dataloader=loader) is False
        
        # After 300+ blocks, should return True
        assert db.should_score(block=500, dataloader=loader) is True

    def test_insert_challenge_and_responses(self, temp_db_path):
        """Test inserting challenge and responses."""
        db = ResponseDatabase(db_path=temp_db_path)
        sample = MockSample()
        miners_data = [
            MinerData(
                uid=i,
                hotkey=f"hotkey_{i}",
                response_time=1.0 + i * 0.1,
                prediction=torch.randn(24, 5, 5)
            )
            for i in range(3)
        ]
        
        db.insert(sample, miners_data)
        
        # Verify challenge was inserted
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM challenges")
        assert cursor.fetchone()[0] == 1
        
        # Verify responses were inserted
        cursor.execute("SELECT COUNT(*) FROM responses")
        assert cursor.fetchone()[0] == 3
        
        conn.close()

    def test_score_and_prune_no_challenges(self, temp_db_path):
        """Test score_and_prune with no challenges."""
        db = ResponseDatabase(db_path=temp_db_path)
        loader = Mock()
        loader.mechanism = MechanismType.ERA5
        loader.get_last_available = Mock(return_value=Mock(timestamp=Mock(return_value=1000000.0)))
        score_func = Mock()
        
        db.score_and_prune(loader, score_func)
        
        # Should not call score_func
        score_func.assert_not_called()

    def test_score_and_prune_with_challenge(self, temp_db_path):
        """Test score_and_prune with a challenge."""
        db = ResponseDatabase(db_path=temp_db_path)
        
        # Insert a challenge
        # end_timestamp should be 24 hours (86400 seconds) after start, not 24 seconds
        sample = MockSample(
            start_timestamp=1000000.0,
            end_timestamp=1000000.0 + (24 * 3600),  # 24 hours later
            output_data=torch.randn(24, 5, 5)
        )
        miners_data = [
            MinerData(
                uid=0,
                hotkey="hotkey_0",
                response_time=1.0,
                prediction=torch.randn(24, 5, 5)
            )
        ]
        db.insert(sample, miners_data)
        
        # Setup loader
        loader = Mock()
        loader.mechanism = MechanismType.ERA5
        # Make sure latest_available is after end_timestamp so the challenge can be scored
        loader.get_last_available = Mock(return_value=Mock(timestamp=Mock(return_value=1000000.0 + (25 * 3600))))
        loader.get_output = Mock(return_value=torch.randn(24, 5, 5))
        loader.sample_cls = MockSample
        
        score_func = Mock()
        
        db.score_and_prune(loader, score_func)
        
        # Should call score_func
        score_func.assert_called_once()

    def test_prune_hotkeys(self, temp_db_path):
        """Test pruning hotkeys."""
        db = ResponseDatabase(db_path=temp_db_path)
        
        # Insert some responses
        sample = MockSample()
        miners_data = [
            MinerData(
                uid=0,
                hotkey="hotkey_to_prune",
                response_time=1.0,
                prediction=torch.randn(24, 5, 5)
            ),
            MinerData(
                uid=1,
                hotkey="hotkey_to_keep",
                response_time=1.0,
                prediction=torch.randn(24, 5, 5)
            )
        ]
        db.insert(sample, miners_data)
        
        # Prune one hotkey
        db.prune_hotkeys(["hotkey_to_prune"])
        
        # Verify only one response remains
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM responses")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("SELECT miner_hotkey FROM responses")
        assert cursor.fetchone()[0] == "hotkey_to_keep"
        
        conn.close()


class TestSerializeDeserialize:
    """Tests for serialize and deserialize functions."""

    def test_serialize_tensor(self, sample_tensor):
        """Test serializing a tensor."""
        result = serialize(sample_tensor)
        assert isinstance(result, str)
        assert result != '[]'

    def test_serialize_none(self):
        """Test serializing None."""
        result = serialize(None)
        assert result == '[]'

    def test_deserialize_tensor(self, sample_tensor):
        """Test deserializing a tensor."""
        serialized = serialize(sample_tensor)
        result = deserialize(serialized)
        assert torch.equal(result, sample_tensor)

    def test_deserialize_none(self):
        """Test deserializing None."""
        result = deserialize(None)
        assert result is None

    def test_serialize_deserialize_roundtrip(self, sample_tensor):
        """Test roundtrip serialization."""
        serialized = serialize(sample_tensor)
        deserialized = deserialize(serialized)
        assert torch.equal(deserialized, sample_tensor)

    def test_serialize_numpy_array(self):
        """Test serializing numpy array."""
        arr = np.array([1, 2, 3])
        result = serialize(arr)
        assert isinstance(result, str)
        deserialized = deserialize(result)
        assert np.array_equal(deserialized.numpy(), arr)


class TestGetColumnNames:
    """Tests for get_column_names function."""

    def test_get_column_names(self, temp_db_path):
        """Test getting column names from a table."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        conn.commit()
        
        columns = get_column_names(cursor, "test_table")
        assert 'id' in columns
        assert 'name' in columns
        
        conn.close()

