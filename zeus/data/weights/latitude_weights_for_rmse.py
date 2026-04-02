
import numpy as np

# ============================================================================
# Original Copyright 2023 DeepMind Technologies Limited.
# 
# The code in this file contains adaptations from the Google DeepMind GraphCast repository,
# specifically the area-weighted latitude loss calculations from `losses.py`.
# Check https://github.com/google-deepmind/graphcast/blob/main/graphcast/losses.py
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: Translated original xarray-based loss calculations into 
# pure NumPy/PyTorch for use in this Bittensor subnet.
# Original source: https://github.com/google-deepmind/graphcast/blob/main/graphcast/losses.py
# ============================================================================

"""
Calculating weights based on latitude is a simple way to approximate the area of each grid cell.

Given latitudes [-89.25, -89.0, -88.75, ..., 88.75, 89.0, 89.25] (d_lat = 0.25)
Each point with `lat` value represents a sphere slice between
`lat - d_lat/2` and `lat + d_lat/2`, and the area of this slice would be
proportional to:
`sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)`, and
we can simply omit the term `2 * sin(d_lat/2)` which is just a constant
that cancels during normalization.

When latitude values fall exactly at the poles.
For example: [-90, -89.25, -89.0, ..., 89.0, 89.25, 90]) (d_lat = 0.25)
Each point with `lat` value again represents a sphere slice between 
`lat - d_lat/2` and `lat + d_lat/2`, except for the points at the poles, 
that represent a slice between `90 - d_lat/2` and `90` or, `-90` and  `-90 + d_lat/2`.
The areas of the first type of point are still proportional to:
* sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)
but for the points at the poles now is:
* sin(90) - sin(90 - d_lat/2) = 2 * sin(d_lat/4) ^ 2
(omitting the common factor of 2 which will be absorbed by the normalization).

It can be shown via a limit, or simple geometry, that in the small angles
regime, the proportion of area per pole-point is equal to 1/8th
the proportion of area covered by each of the nearest non-pole point.
"""

def calculate_weights():
    """"
    Calculating and saving the weights for 721x1440 grid to be used during rmse calculation.
    """
    delta_latitude = 0.25
    latitudes = np.arange(-90, 90 + delta_latitude, delta_latitude)
    print(latitudes[0], latitudes[-1])
    weights = np.cos(np.deg2rad(latitudes)) * np.sin(np.deg2rad(delta_latitude/2))
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    #weights = weights / weights.mean()
    print(f'Weights shape {weights.shape}')
    np.save('zeus/data/weights/latitude_weights_for_rmse.npy', weights)

calculate_weights()
