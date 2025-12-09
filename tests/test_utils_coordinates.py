"""
Unit tests for zeus.utils.coordinates module.
"""
import pytest
import numpy as np
import torch
from zeus.utils.coordinates import (
    bbox_to_str,
    get_bbox,
    slice_bbox,
    get_grid,
    gaussian_grid_sample,
    expand_to_grid,
    interp_coordinates,
)


class TestBboxToStr:
    """Tests for bbox_to_str function."""

    def test_bbox_to_str_basic(self):
        """Test basic bbox to string conversion."""
        bbox = (-10.0, 10.0, -20.0, 20.0)
        result = bbox_to_str(bbox)
        assert isinstance(result, str)
        assert "lat_start" in result.lower() or "-10.00" in result

    def test_bbox_to_str_format(self):
        """Test bbox string format."""
        bbox = (0.0, 1.0, 2.0, 3.0)
        result = bbox_to_str(bbox)
        assert isinstance(result, str)
        assert len(result) > 0


class TestGetBbox:
    """Tests for get_bbox function."""

    def test_get_bbox_3d_tensor(self):
        """Test getting bbox from 3D tensor."""
        # Create a tensor with lat/lon in first two dims
        tensor = torch.zeros(5, 5, 2)
        tensor[:, :, 0] = torch.linspace(-10, 10, 5).unsqueeze(1).repeat(1, 5)  # lat
        tensor[:, :, 1] = torch.linspace(-20, 20, 5).unsqueeze(0).repeat(5, 1)  # lon
        
        result = get_bbox(tensor)
        assert len(result) == 4
        assert isinstance(result[0], float)

    def test_get_bbox_4d_tensor(self):
        """Test getting bbox from 4D tensor."""
        # Create a 4D tensor (time, lat, lon, vars)
        tensor = torch.zeros(2, 5, 5, 2)
        tensor[:, :, :, 0] = torch.linspace(-10, 10, 5).unsqueeze(1).unsqueeze(0).repeat(2, 1, 5)
        tensor[:, :, :, 1] = torch.linspace(-20, 20, 5).unsqueeze(0).unsqueeze(0).repeat(2, 5, 1)
        
        result = get_bbox(tensor)
        assert len(result) == 4


class TestSliceBbox:
    """Tests for slice_bbox function."""

    def test_slice_bbox_basic(self):
        """Test basic bbox slicing."""
        # Create a matrix representing global data
        fidelity = 4
        matrix = torch.zeros(180 * fidelity + 1, 360 * fidelity)
        bbox = (-10.0, 10.0, -20.0, 20.0)
        
        result = slice_bbox(matrix, bbox)
        assert result.shape[0] <= matrix.shape[0]
        assert result.shape[1] <= matrix.shape[1]

    def test_slice_bbox_preserves_other_dims(self):
        """Test that slicing preserves other dimensions."""
        fidelity = 4
        matrix = torch.zeros(180 * fidelity + 1, 360 * fidelity, 3)
        bbox = (-10.0, 10.0, -20.0, 20.0)
        
        result = slice_bbox(matrix, bbox)
        assert result.shape[2] == 3  # Preserve third dimension


class TestGetGrid:
    """Tests for get_grid function."""

    def test_get_grid_basic(self):
        """Test basic grid creation."""
        grid = get_grid(-10.0, 10.0, -20.0, 20.0, fidelity=4)
        assert isinstance(grid, torch.Tensor)
        assert grid.shape[-1] == 2  # lat, lon

    def test_get_grid_shape(self):
        """Test grid shape."""
        grid = get_grid(-1.0, 1.0, -1.0, 1.0, fidelity=4)
        # Should have shape (lat_points, lon_points, 2)
        assert len(grid.shape) == 3
        assert grid.shape[2] == 2

    def test_get_grid_values(self):
        """Test grid values are in correct range."""
        grid = get_grid(-10.0, 10.0, -20.0, 20.0, fidelity=1)
        lats = grid[:, :, 0]
        lons = grid[:, :, 1]
        assert lats.min() >= -10.0
        assert lats.max() <= 10.0
        assert lons.min() >= -20.0
        assert lons.max() <= 20.0


class TestGaussianGridSample:
    """Tests for gaussian_grid_sample function."""

    def test_gaussian_grid_sample_returns_point(self):
        """Test that gaussian_grid_sample returns a point."""
        grid = get_grid(-10.0, 10.0, -20.0, 20.0, fidelity=4)
        result = gaussian_grid_sample(grid)
        assert len(result) == 2
        assert isinstance(result[0], (float, np.floating, torch.Tensor))

    def test_gaussian_grid_sample_in_range(self):
        """Test that sampled point is within grid bounds."""
        grid = get_grid(-10.0, 10.0, -20.0, 20.0, fidelity=4)
        for _ in range(10):  # Test multiple samples
            result = gaussian_grid_sample(grid)
            lat, lon = result
            assert -10.0 <= lat <= 10.0
            assert -20.0 <= lon <= 20.0


class TestExpandToGrid:
    """Tests for expand_to_grid function."""

    def test_expand_to_grid_basic(self):
        """Test basic grid expansion."""
        grid = expand_to_grid(0.0, 0.0, fidelity=4)
        assert isinstance(grid, torch.Tensor)
        assert grid.shape[-1] == 2

    def test_expand_to_grid_shape(self):
        """Test expanded grid shape."""
        grid = expand_to_grid(0.0, 0.0, fidelity=4)
        assert len(grid.shape) == 3
        assert grid.shape[2] == 2

    def test_expand_to_grid_contains_point(self):
        """Test that expanded grid contains the original point."""
        lat, lon = 40.0, -74.0
        grid = expand_to_grid(lat, lon, fidelity=4)
        lats = grid[:, :, 0]
        lons = grid[:, :, 1]
        assert lats.min() <= lat <= lats.max()
        assert lons.min() <= lon <= lons.max()


class TestInterpCoordinates:
    """Tests for interp_coordinates function."""

    def test_interp_coordinates_basic(self):
        """Test basic coordinate interpolation."""
        input_tensor = torch.randn(24, 5, 5)  # time, lat, lon
        # Grid must match the lat/lon dimensions of the input tensor (5, 5)
        # For 5 points: (end - start) * fidelity + 1 = 5, so (end - start) * 1 + 1 = 5, so (end - start) = 4
        # We want symmetric around 0, so -2 to 2 for both lat and lon gives us a 5x5 grid
        grid = get_grid(-2.0, 2.0, -2.0, 2.0, fidelity=1)  # This gives 5x5 grid matching input tensor
        result = interp_coordinates(input_tensor, grid, 0.0, 0.0)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 24  # Preserve time dimension

    def test_interp_coordinates_shape(self):
        """Test interpolation output shape."""
        input_tensor = torch.randn(24, 5, 5)  # time, lat, lon
        # Grid must match the lat/lon dimensions of the input tensor (5, 5)
        # Create a 5x5 grid to match the input tensor dimensions
        grid = get_grid(-2.0, 2.0, -2.0, 2.0, fidelity=1)  # This gives 5x5 grid
        result = interp_coordinates(input_tensor, grid, 0.0, 0.0)
        # Should return (time, 1, 1)
        assert result.shape == (24, 1, 1)

