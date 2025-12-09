# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import Tuple, Union, List
import numpy as np
import math
from check_shapes import check_shapes
import torch
import torch.nn.functional as F

from zeus.validator.constants import ERA5_AREA_SAMPLE_RANGE

def bbox_to_str(bbox: Tuple[Union[float, np.number]]) -> str:
    """
    Convert the bounding box to a string.
    """
    return f"[lat_start, lat_end, lon_start, lon_end]=[{float(bbox[0]):.2f}, {float(bbox[1]):.2f}, {float(bbox[2]):.2f}, {float(bbox[3]):.2f}]"


def get_bbox(
    tensor: Union[np.ndarray, torch.Tensor],
) -> Tuple[float, float, float, float]:
    """
    Returns the bounding box of the given tensor, which should be 3 or 4 dimensional (optional time, lat, lon, variables).
    Variables should be in order of latitude, longitude, and then the rest.

    Returns:
    - bbox (Tuple[float, float, float, float]): The bounding box of the tensor: Latitude start, latitude end, longitude start, longitude end.
    """
    if tensor.ndim == 4:
        # so it works regardless of time dimension
        tensor = tensor[0]
    lat_start = tensor[0, 0, 0].item()
    lat_end = tensor[-1, 0, 0].item()
    lon_start = tensor[0, 0, 1].item()
    lon_end = tensor[0, -1, 1].item()
    return lat_start, lat_end, lon_start, lon_end

def slice_bbox(
    matrix: Union[np.ndarray, torch.Tensor], bbox: Tuple[float, float, float, float], lat_dim:int = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Slice the matrix to the given lat-lon bounding box. This assumes that the matrix is of shape (180 * fidelity + 1, 360 * fidelity, ...).
    NOTE: it is also assumed that coordinates are in the range of -90 to 90 for latitude and -180 to 179.75 for longitude.

    Lat_dim can optionally be used to specify the dimension of the latitude data (defaults to 0)
     longitude dimension is assumed to be lat_dim + 1.
    """

    fidelity = matrix.shape[lat_dim + 1] // 360

    lat_start, lat_end, lon_start, lon_end = bbox
    lat_start_idx = int((90 + lat_start) * fidelity)
    lat_end_idx = int((90 + lat_end) * fidelity)
    lon_start_idx = int((180 + lon_start) * fidelity)
    lon_end_idx = int((180 + lon_end) * fidelity)

    # slice across specified dimensions only
    sl = [slice(None)] * matrix.ndim
    sl[lat_dim] = slice(lat_start_idx, lat_end_idx + 1)
    sl[lat_dim + 1] = slice(lon_start_idx, lon_end_idx + 1)

    return matrix[tuple(sl)]
    
def get_grid(
    lat_start: float,
    lat_end: float,
    lon_start: float,
    lon_end: float,
    fidelity: int = 4,
) -> torch.Tensor:
    """
    Get a grid of lat-lon points in the given bounding box.
    Args:
        lat_start: The start latitude.
        lat_end: The end latitude.
        lon_start: The start longitude.
        lon_end: The end longitude.
        fidelity: The fidelity of the grid.

    Returns:
        torch.Tensor: The grid of lat-lon points. Shape: (lat, lon, 2)
    """
    return torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(start, end, int((end - start) * fidelity) + 1)
                for start, end in [(lat_start, lat_end), (lon_start, lon_end)]
            ],
            indexing="ij",
        ),
        dim=-1,
    )  # (lat, lon, 2)

@check_shapes(
    "grid: [lat, lon, 2]",
    "return: [2]",
)
def gaussian_grid_sample(grid: Union[torch.Tensor, np.ndarray], stds_in_radius = 3) -> Tuple[float, float]:
    """
    Sample a random grid point from the given grid using a Gaussian distribution.
    Args:
        grid: The grid to sample from.
        stds_in_radius: The number of standard deviations as a factor of the radius.

    Returns:
        Tuple[float, float]: The sampled grid point.
    """
    lat, lon = grid.shape[:2]

    while True:
        # 91% within 2 stds on both axis
        i = round(np.random.normal(lat / 2, (lat / 2 / stds_in_radius) ** 2))
        j = round(np.random.normal(lon / 2, (lon / 2 / stds_in_radius) ** 2))

        if 0 <= i < lat and 0 <= j < lon:
            return grid[i, j]


def expand_to_grid(
    lat: float, lon: float, fidelity: float = 4, min_size = ERA5_AREA_SAMPLE_RANGE[0],
) -> torch.Tensor:
    """
    Get a grid of lat-lon points for a single location based on the specified fidelity (degree).
    If the specified lat or lon coordinate are exactly on a grid point that dimension will be min_size + 1,
    otherwise it will be of min_size.
    The output grid will have the same shape structure as the get_grid function.
    """
    lat_start = math.floor(lat * fidelity) / fidelity
    lat_end = math.ceil(lat * fidelity) / fidelity

    if lat_start == lat_end:
        lat_start -= 1 / fidelity
        lat_end += 1 / fidelity

    lon_start = math.floor(lon * fidelity) / fidelity
    lon_end = math.ceil(lon * fidelity) / fidelity

    if lon_start == lon_end:
        lon_start -= 1 / fidelity
        lon_end += 1 / fidelity

    expand = max(0, (min_size - 2) // 2) / fidelity
    return get_grid(
        lat_start - expand, lat_end + expand, 
        lon_start - expand, lon_end + expand,
        fidelity=fidelity
    )

@check_shapes(
    "input: [time, lat, lon]",
    "grid: [lat, lon, 2]",
    "return: [time, 1, 1]",
)
def interp_coordinates(
        input: torch.Tensor,
        grid: torch.Tensor,
        to_lat: float,
        to_lon: float,
) -> torch.Tensor:
    """
    Interpolate the input tensor to the given latitude and longitude.
    Args:
        input: The input tensor to interpolate.
        grid: The grid to interpolate to.
        to_lat: The latitude to interpolate to.
        to_lon: The longitude to interpolate to.

    Returns:
        torch.Tensor: The interpolated tensor. Shape: (time, 1, 1)
    """
    coords = torch.tensor([to_lat, to_lon])
    interp_grid = (coords - grid[0,0]) / (grid[-1,-1] - grid[0, 0])
    interp_grid = (interp_grid * 2 - 1).repeat(input.size(0), 1, 1, 1) # time, 1, 1, 2

    return F.grid_sample(input.unsqueeze(1), grid=interp_grid,align_corners=True).squeeze(-1)