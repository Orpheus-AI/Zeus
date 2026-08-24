from typing import Optional, Tuple, Type
from pathlib import Path

import numpy as np
import torch
import time
import bittensor as bt
from zeus.utils.coordinates import get_bbox, get_grid, slice_bbox
from zeus.utils.time import to_timestamp
from zeus.protocol import TimePredictionSynapse, PredictionSynapse
from zeus import __version__ as zeus_version
from zeus.validator.challenge_spec import make_state_key
from zeus.validator.constants import (
    DEFAULT_STEP_SIZE,
    WIND_SCALARS_PATH,
    SOLAR_SCALARS_PATH,
    TEMPERATURE_SCALARS_PATH,
)
from zeus.utils.region_mask import REGION_CONFIGS, build_geographic_weights

_SCALAR_CACHE: dict[Path, dict[str, np.ndarray]] = {}


def _load_scalar_npz(path: Path) -> Optional[dict[str, np.ndarray]]:
    """Load scalars npz once. Keys: scalars (lat, lon), lats (lat,), lons (lon,)."""
    if path in _SCALAR_CACHE:
        return _SCALAR_CACHE[path]
    if not path.exists():
        return None
    with np.load(path) as data:
        payload = {
            "scalars": np.asarray(data["scalars"]),
            "lats": np.asarray(data["lats"], dtype=np.float64),
            "lons": np.asarray(data["lons"], dtype=np.float64),
        }
    if payload["scalars"].shape != (len(payload["lats"]), len(payload["lons"])):
        return None
    _SCALAR_CACHE[path] = payload
    return payload


def _sliced_scalar_tensor(path: Path, x_grid: torch.Tensor) -> Optional[torch.Tensor]:
    """Crop full ERA5 scalar field to x_grid bbox; same shape as old_europe_weight."""
    payload = _load_scalar_npz(path)
    if payload is None:
        return None
    lats, lons = payload["lats"], payload["lons"]
    if not (np.all(np.diff(lats) > 0) and np.all(np.diff(lons) > 0)):
        return None
    cropped = slice_bbox(payload["scalars"], get_bbox(x_grid))
    if cropped.shape != tuple(x_grid.shape[:2]):
        return None
    return torch.as_tensor(np.ascontiguousarray(cropped), dtype=torch.float32)


class Era5Sample:

    def __init__(
        self,
        start_timestamp: float,
        end_timestamp: float,
        lat_start: float,
        lat_end: float,
        lon_start: float,
        lon_end: float,
        variable: str,
        query_timestamp: Optional[int] = None,
        output_data: Optional[torch.Tensor] = None,
        predict_hours: Optional[int] = None,
        step_size: int = DEFAULT_STEP_SIZE,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
    ):
        """
        Create a datasample, either containing actual data or representing a database entry.
        """
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lon_start = lon_start
        self.lon_end = lon_end

        self.variable = variable
        self.query_timestamp = query_timestamp or round(time.time())

        self.output_data = output_data
        self.predict_hours = predict_hours
        self.step_size = step_size

        self.start_offset = start_offset
        self.end_offset = end_offset

        self.x_grid = get_grid(lat_start, lat_end, lon_start, lon_end)
        # Pre-cutoff / fallback: Europe+Germany axis-aligned box weights.
        self.old_europe_weight = build_geographic_weights(self.x_grid, REGION_CONFIGS)

        self.wind_scalar = _sliced_scalar_tensor(WIND_SCALARS_PATH, self.x_grid)
        self.solar_scalar = _sliced_scalar_tensor(SOLAR_SCALARS_PATH, self.x_grid)
        self.temperature_scalar = _sliced_scalar_tensor(TEMPERATURE_SCALARS_PATH, self.x_grid)

        if self.wind_scalar is None:
            raise FileNotFoundError(f"Missing wind scalars at {WIND_SCALARS_PATH}")
        if self.solar_scalar is None:
            raise FileNotFoundError(f"Missing solar scalars at {SOLAR_SCALARS_PATH}")
        if self.temperature_scalar is None:
            raise FileNotFoundError(f"Missing temperature scalars at {TEMPERATURE_SCALARS_PATH}")

        if output_data is not None:
            self.predict_hours = output_data.shape[0]
        elif predict_hours is None:
            raise ValueError("Either output data or predict hours must be provided.")

    @property
    def state_key(self) -> str:
        if self.start_offset is None or self.end_offset is None:
            raise ValueError("start_offset and end_offset must be set to derive state_key")
        return make_state_key(self.variable, self.start_offset, self.end_offset)


    def get_bbox(self) -> Tuple[float]:
        return self.lat_start, self.lat_end, self.lon_start, self.lon_end

    def build_synapse(self, synapse_cls: Type[PredictionSynapse]) -> PredictionSynapse:
        kwargs = {
            "version": zeus_version,
            "start_time": self.start_timestamp,
            "end_time": self.end_timestamp,
            "requested_hours": self.predict_hours,
            "variable": self.variable,
            "step_size": self.step_size,
            "latitude_start": self.lat_start,
            "latitude_end": self.lat_end,
            "longitude_start": self.lon_start,
            "longitude_end": self.lon_end,
        }

        if issubclass(synapse_cls, TimePredictionSynapse):
            bt.logging.info(
                f"predict_hours: {self.predict_hours} "
                f"start_time: {to_timestamp(self.start_timestamp)} "
                f"end_time: {to_timestamp(self.end_timestamp)} "
                f"step_size: {self.step_size}"
            )
            kwargs["locations"] = self.x_grid.tolist()

        return synapse_cls(**kwargs)
    
    def __str__(self) -> str:
        return f'{self.lat_start}_{self.lat_end}_{self.lon_start}_{self.lon_end}_{self.variable}_{self.start_timestamp}_{self.end_timestamp}_{self.predict_hours}'
