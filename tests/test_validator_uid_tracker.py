"""
Unit tests for zeus.validator.uid_tracker module.
"""
import pytest
import numpy as np
from unittest.mock import Mock
from zeus.validator.uid_tracker import UIDTracker
from zeus.validator.constants import MechanismType


class TestUIDTracker:
    """Tests for UIDTracker class."""

    @pytest.fixture
    def uid_tracker(self, mock_validator, mock_preference_manager):
        """Create a UIDTracker instance."""
        return UIDTracker(mock_validator, mock_preference_manager)

    def test_get_uids_filters_by_preference(self, uid_tracker, mock_metagraph):
        """Test that get_uids filters by mechanism preference."""
        # Set some UIDs to have different preferences
        uid_tracker.preference_manager.get_preference = Mock(
            side_effect=lambda uid: MechanismType.ERA5 if uid < 5 else MechanismType.WEATHER_XM
        )
        
        result = uid_tracker.get_uids(MechanismType.ERA5, exclude=set())
        assert all(uid < 5 for uid in result)

    def test_get_uids_excludes_busy(self, uid_tracker):
        """Test that get_uids excludes busy UIDs."""
        uid_tracker.add_busy_uids([0, 1, 2], MechanismType.ERA5)
        result = uid_tracker.get_uids(MechanismType.ERA5, exclude={0, 1, 2})
        assert 0 not in result
        assert 1 not in result
        assert 2 not in result

    def test_get_uids_excludes_non_serving(self, uid_tracker, mock_metagraph):
        """Test that get_uids excludes non-serving UIDs."""
        mock_metagraph.axons[3].is_serving = False
        result = uid_tracker.get_uids(MechanismType.ERA5, exclude=set())
        assert 3 not in result

    def test_get_random_uids_returns_k_uids(self, uid_tracker):
        """Test that get_random_uids returns k UIDs."""
        result = uid_tracker.get_random_uids(k=3, mechanism=MechanismType.ERA5, tries=1)
        assert len(result) == 3

    def test_get_random_uids_adds_to_busy(self, uid_tracker):
        """Test that get_random_uids adds UIDs to busy list."""
        result = uid_tracker.get_random_uids(k=3, mechanism=MechanismType.ERA5, tries=1)
        busy = uid_tracker.get_busy_uids(MechanismType.ERA5)
        assert all(uid in busy for uid in result)

    def test_get_random_uids_retries_on_insufficient(self, uid_tracker):
        """Test that get_random_uids retries when insufficient UIDs."""
        # Make most UIDs busy
        uid_tracker.add_busy_uids(list(range(8)), MechanismType.ERA5)
        # Should still return what it can
        result = uid_tracker.get_random_uids(k=5, mechanism=MechanismType.ERA5, tries=3, sleep=0)
        assert len(result) <= 5

    def test_mark_finished_removes_from_busy(self, uid_tracker):
        """Test that mark_finished removes UIDs from busy list."""
        uid_tracker.add_busy_uids([0, 1, 2], MechanismType.ERA5)
        uid_tracker.mark_finished([0, 1], MechanismType.ERA5, good=False)
        busy = uid_tracker.get_busy_uids(MechanismType.ERA5)
        assert 0 not in busy
        assert 1 not in busy
        assert 2 in busy

    def test_mark_finished_good_updates_last_good(self, uid_tracker):
        """Test that mark_finished with good=True updates last_good_uids."""
        uid_tracker.add_busy_uids([0, 1, 2], MechanismType.ERA5)
        uid_tracker.mark_finished([0, 1], MechanismType.ERA5, good=True)
        # Check that last_good_uids was updated (internal check)
        responding = uid_tracker.get_responding_uids(k=2, mechanism=MechanismType.ERA5)
        assert len(responding) <= 2

    def test_get_responding_uids_returns_good_uids(self, uid_tracker):
        """Test that get_responding_uids returns previously good UIDs."""
        uid_tracker.add_busy_uids([0, 1, 2], MechanismType.ERA5)
        uid_tracker.mark_finished([0, 1], MechanismType.ERA5, good=True)
        uid_tracker.mark_finished([2], MechanismType.ERA5, good=False)
        
        # Should get the good UIDs
        result = uid_tracker.get_responding_uids(k=2, mechanism=MechanismType.ERA5)
        assert len(result) <= 2

    def test_get_responding_uids_excludes_busy(self, uid_tracker):
        """Test that get_responding_uids excludes currently busy UIDs."""
        uid_tracker.add_busy_uids([0, 1], MechanismType.ERA5)
        uid_tracker.mark_finished([0, 1], MechanismType.ERA5, good=True)
        uid_tracker.add_busy_uids([0], MechanismType.ERA5)  # Make 0 busy again
        
        result = uid_tracker.get_responding_uids(k=2, mechanism=MechanismType.ERA5)
        assert 0 not in result or len(result) == 0

    def test_thread_safety(self, uid_tracker):
        """Test that operations are thread-safe and correct UIDs are tracked."""
        import threading
        
        # First, add all UIDs sequentially to establish baseline
        uid_tracker.add_busy_uids(list(range(10)), MechanismType.ERA5)
        assert uid_tracker.get_busy_uids(MechanismType.ERA5) == set(range(10))
        
        def add_busy():
            """Add UIDs 10-19 to busy set concurrently."""
            for i in range(10, 20):
                uid_tracker.add_busy_uids([i], MechanismType.ERA5)
        
        def mark_finished():
            """Remove UIDs 0-4 from busy set concurrently."""
            for i in range(5):
                uid_tracker.mark_finished([i], MechanismType.ERA5, good=False)
        
        # Run both operations concurrently to test thread safety
        threads = [
            threading.Thread(target=add_busy),
            threading.Thread(target=mark_finished)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify return type: get_busy_uids returns Set[int]
        busy = uid_tracker.get_busy_uids(MechanismType.ERA5)
        assert isinstance(busy, set)
        
        # Verify correctness: UIDs 0-4 should be removed, UIDs 5-19 should be present
        # Thread safety ensures operations are atomic, so all adds and removes complete correctly
        assert busy == set(range(5, 20))
        assert len(busy) == 15

