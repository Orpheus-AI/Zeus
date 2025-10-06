from typing import Optional, Tuple
import torch
import time

from zeus.utils.coordinates import get_grid
from zeus.protocol import TimePredictionSynapse
from zeus import __version__ as zeus_version


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
        om_baseline: Optional[torch.Tensor] = None,
        ifs_hres_baseline: Optional[torch.Tensor] = None
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
       
        self.x_grid = get_grid(lat_start, lat_end, lon_start, lon_end)

        if output_data is not None:
            self.predict_hours = output_data.shape[0]
        elif predict_hours is None:
            raise ValueError("Either output data or predict hours must be provided.")
        
        # Usually set through OpenMeteoLoader or from database
        self.om_baseline = om_baseline
        self.ifs_hres_baseline = ifs_hres_baseline
        

    def get_bbox(self) -> Tuple[float]:
        return self.lat_start, self.lat_end, self.lon_start, self.lon_end

    def get_synapse(self) -> TimePredictionSynapse:
        """
        Converts the sample to a synapse which miners can predict on.
        Note that the output data is NOT set in this synapse.
        """
        return TimePredictionSynapse(
            version=zeus_version,
            locations=self.x_grid.tolist(),
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            requested_hours=self.predict_hours,
            variable=self.variable
        )