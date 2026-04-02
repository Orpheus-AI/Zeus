from typing import Optional, Tuple, Type
import torch
import time
import bittensor as bt
from zeus.utils.coordinates import get_grid
from zeus.utils.time import to_timestamp
from zeus.protocol import HashedTimePredictionSynapse, TimePredictionSynapse, PredictionSynapse
from zeus import __version__ as zeus_version
from zeus.validator.constants import DEFAULT_STEP_SIZE
from zeus.validator.challenge_spec import make_state_key
from zeus.utils.region_mask import europe_mask_for_grid
from zeus.validator.constants import EUROPE_WEIGHT, DEFAULT_STEP_SIZE

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
        europe_mask = europe_mask_for_grid(self.x_grid)
        self.europe_weight = torch.where(europe_mask == 1, EUROPE_WEIGHT, 1)

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
