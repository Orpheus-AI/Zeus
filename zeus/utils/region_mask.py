"""Region masks and geographic weighting for lat-lon grids."""

from dataclasses import dataclass

import torch

from zeus.validator.constants import (
    EUROPE_WEIGHT,
    EUROPE_LATITUDE_RANGE,
    EUROPE_LONGITUDE_RANGE,
    GERMANY_WEIGHT,
    GERMANY_LATITUDE_RANGE,
    GERMANY_LONGITUDE_RANGE,
)


@dataclass(frozen=True)
class RegionConfig:
    name: str
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    weight: float

EUROPE_REGION_CONFIG: RegionConfig = RegionConfig(
        name="Europe",
        lat_range=EUROPE_LATITUDE_RANGE,
        lon_range=EUROPE_LONGITUDE_RANGE,
        weight=EUROPE_WEIGHT,
    )
GERMANY_REGION_CONFIG: RegionConfig = RegionConfig(
        name="Germany",
        lat_range=GERMANY_LATITUDE_RANGE,
        lon_range=GERMANY_LONGITUDE_RANGE,
        weight=GERMANY_WEIGHT,
    )
# TODO: Remove this once we have evaluated all the challenges before the update
OLD_REGION_CONFIGS: list[RegionConfig] = [
    EUROPE_REGION_CONFIG,
]

REGION_CONFIGS: list[RegionConfig] = [
    EUROPE_REGION_CONFIG,
    GERMANY_REGION_CONFIG,
]


def region_mask_for_grid(
    grid: torch.Tensor,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> torch.Tensor:
    """
    Return a mask of shape (n_lat, n_lon) with 1 inside the axis-aligned lat/lon box, 0 otherwise.
    grid must have shape (n_lat, n_lon, 2) with grid[..., 0] = lat, grid[..., 1] = lon.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    in_lat = (grid[..., 0] >= lat_min) & (grid[..., 0] <= lat_max)
    in_lon = (grid[..., 1] >= lon_min) & (grid[..., 1] <= lon_max)
    return (in_lat & in_lon).to(torch.float32)



def build_geographic_weights(
    grid: torch.Tensor,
    configs: list[RegionConfig] | None = None,
    default_weight: float = 1.0,
) -> torch.Tensor:
    """
    Per-cell multipliers: start at default_weight. For each region in order, cells inside that
    region's box are set to max(current cell, region.weight); other cells are unchanged.
    With [Europe, Germany] and Germany ⊂ Europe, Germany gets GERMANY_WEIGHT and the rest of
    Europe gets EUROPE_WEIGHT (not GERMANY_WEIGHT).
    """
    if configs is None:
        configs = REGION_CONFIGS
    weights = torch.full(
        grid.shape[:-1],
        default_weight,
        dtype=torch.float32,
        device=grid.device,
    )
    for region in configs:
        mask = region_mask_for_grid(grid, region.lat_range, region.lon_range)
        max_weight = torch.maximum(weights, torch.tensor(region.weight, device=weights.device))
        weights = torch.where(mask == 1, max_weight, weights)
    return weights


if __name__ == "__main__":
    from zeus.utils.coordinates import get_grid

    grid = get_grid(-90, 90, -180, 179.75)
    mask = region_mask_for_grid(grid, EUROPE_LATITUDE_RANGE, EUROPE_LONGITUDE_RANGE)

    lat_min, lat_max = EUROPE_LATITUDE_RANGE
    lon_min, lon_max = EUROPE_LONGITUDE_RANGE
    ones = mask == 1
    lats = grid[..., 0][ones]
    lons = grid[..., 1][ones]
    assert (lats >= lat_min).all() and (lats <= lat_max).all(), "lat out of range"
    assert (lons >= lon_min).all() and (lons <= lon_max).all(), "lon out of range"
    in_box = (
        (grid[..., 0] >= lat_min)
        & (grid[..., 0] <= lat_max)
        & (grid[..., 1] >= lon_min)
        & (grid[..., 1] <= lon_max)
    )
    assert (mask[in_box] == 1).all(), "inside box but not 1"
    print("mask.shape", mask.shape, "ones", mask.sum().item())
