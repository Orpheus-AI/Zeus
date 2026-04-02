"""Region masks for lat-lon grids (e.g. Europe)."""


from zeus.utils.coordinates import get_grid
import torch
from zeus.validator.constants import EUROPE_LATITUDE_RANGE, EUROPE_LONGITUDE_RANGE, EUROPE_WEIGHT


def europe_mask_for_grid(grid):
    """
    Return a mask of shape (n_lat, n_lon) with 1 where (lat, lon) is in Europe and 0 otherwise.
    grid must have shape (n_lat, n_lon, 2) with grid[..., 0] = lat, grid[..., 1] = lon.
    """
    lat_min, lat_max = EUROPE_LATITUDE_RANGE
    lon_min, lon_max = EUROPE_LONGITUDE_RANGE
    in_lat = (grid[..., 0] >= lat_min) & (grid[..., 0] <= lat_max)
    in_lon = (grid[..., 1] >= lon_min) & (grid[..., 1] <= lon_max)
    mask = (in_lat & in_lon).to(torch.float32)
    return mask


if __name__ == "__main__":
    grid = get_grid(-90, 90, -180, 179.75)
    mask = europe_mask_for_grid(grid)
    lat_min, lat_max = EUROPE_LATITUDE_RANGE
    lon_min, lon_max = EUROPE_LONGITUDE_RANGE
    # Sanity: every 1 must be inside the box
    ones = (mask == 1)
    lats = grid[..., 0][ones]
    lons = grid[..., 1][ones]
    assert (lats >= lat_min).all() and (lats <= lat_max).all(), "lat out of range"
    assert (lons >= lon_min).all() and (lons <= lon_max).all(), "lon out of range"
    # Sanity: every point inside the box must be 1
    in_box = (grid[..., 0] >= lat_min) & (grid[..., 0] <= lat_max) & (grid[..., 1] >= lon_min) & (grid[..., 1] <= lon_max)
    assert (mask[in_box] == 1).all(), "inside box but not 1"
    print("mask.shape", mask.shape, "ones", mask.sum().item())