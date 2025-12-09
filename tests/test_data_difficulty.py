"""
Unit tests for zeus.data.difficulty module.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from zeus.data.difficulty import DifficultyLoader
from zeus.data.base.sample import BaseSample
from zeus.validator.constants import MechanismType


class TestDifficultyLoader:
    """Tests for DifficultyLoader class."""

    @pytest.fixture
    def temp_data_folder(self):
        """Create a temporary data folder with difficulty matrices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create weights folder inside temp directory
            weights_folder = Path(tmpdir) / "weights"
            weights_folder.mkdir()
            
            # Create some difficulty matrices for different months
            for month in [1, 6, 12]:
                matrix = np.random.rand(180, 360).astype(np.float32)
                file_path = weights_folder / f"difficulty_{month}.npy"
                np.save(file_path, matrix)
            
            # Yield the parent directory (where difficulty.py would be)
            yield Path(tmpdir)

    @pytest.fixture
    def mock_sample(self):
        """Create a mock sample for testing."""
        sample = Mock(spec=BaseSample)
        sample.start_timestamp = 1000000.0  # January 1, 1970 00:00:00 (approximately)
        sample.get_bbox = Mock(return_value=(-10.0, 10.0, -20.0, 20.0))
        return sample

    def test_difficulty_loader_init_loads_matrices(self, temp_data_folder):
        """Test that DifficultyLoader loads difficulty matrices on initialization."""
        # Mock os.path.abspath to return a path in the temp folder
        mock_file_path = temp_data_folder / "difficulty.py"
        
        with patch('zeus.data.difficulty.os.path.abspath', return_value=str(mock_file_path)):
            # Create loader - it should find the matrices in temp_data_folder/weights
            loader = DifficultyLoader(data_folder="weights")
            
            assert len(loader.difficulty_matrices) == 3
            assert 1 in loader.difficulty_matrices
            assert 6 in loader.difficulty_matrices
            assert 12 in loader.difficulty_matrices

    def test_difficulty_loader_init_no_matrices_raises_error(self):
        """Test that DifficultyLoader raises error when no matrices found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_folder = Path(tmpdir) / "weights"
            weights_folder.mkdir()
            # Don't create any .npy files
            
            mock_file_path = Path(tmpdir) / "difficulty.py"
            with patch('zeus.data.difficulty.os.path.abspath', return_value=str(mock_file_path)):
                with pytest.raises(AssertionError) as exc_info:
                    DifficultyLoader(data_folder="weights")
                assert "No difficulty matrices found" in str(exc_info.value)

    def test_get_difficulty_grid_uses_correct_month(self, temp_data_folder, mock_sample):
        """Test that get_difficulty_grid uses the correct month's matrix."""
        mock_file_path = temp_data_folder / "difficulty.py"
        with patch('zeus.data.difficulty.os.path.abspath', return_value=str(mock_file_path)):
            loader = DifficultyLoader(data_folder="weights")
            
            # Mock slice_bbox to return a subset
            with patch('zeus.data.difficulty.slice_bbox') as mock_slice:
                expected_grid = np.random.rand(5, 5)
                mock_slice.return_value = expected_grid
                
                result = loader.get_difficulty_grid(mock_sample)
                
                # Verify slice_bbox was called with the correct matrix and bbox
                assert mock_slice.called
                call_args = mock_slice.call_args
                # The first argument should be one of the difficulty matrices
                # Check if the array matches any of the stored matrices
                called_matrix = call_args[0][0]
                assert any(np.array_equal(called_matrix, matrix) for matrix in loader.difficulty_matrices.values())
                assert call_args[0][1] == mock_sample.get_bbox()
                assert np.array_equal(result, expected_grid)