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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Any, Callable, List

import numpy as np
import torch
from check_shapes import check_shapes

@dataclass
class VariableConverter(ABC):
    """
    Utilities to convert OpenMeteo's variables and units to normal ERA5 
    and vice versa
    """

    # their main ERA5 representation, used as key throughout
    data_var: str

    # OpenMeteo variable name
    om_name: Union[str, List[str]]
    # Abbreviated ERA5 name, which is how they are saved internally in NC files
    short_code: str
    # Metric SI unit as string
    unit: str

    def era5_to_om(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return data

    def om_to_era5(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return data
    

class TemperatureConverter(VariableConverter):

    def era5_to_om(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return data - 273.15
    
    def om_to_era5(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return data + 273.15


class PrecipitationConverter(VariableConverter):

    def era5_to_om(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return data * 1e3
    
    def om_to_era5(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return data / 1e3
    

class WindConverter(VariableConverter, ABC):

    def era5_to_om(self, data: Any) -> Any:
        raise NotImplementedError

    @check_shapes(
        "data: [batch..., n]",
        "return: [batch...]",
    )
    def om_to_era5(self, data: Union[np.ndarray, torch.Tensor], trigeometry: Callable) -> Union[np.ndarray, torch.Tensor]:
        """
        OpenMeteo only provides overall wind speed (km/h) and wind direction at 100 meters.
        So we convert this to eastern wind at a 100 meters (m/s). We average altitudes.
        See: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398

        data: Any array/tensor with last dimension being the variables (in order!):
            "wind speed 100m", "wind direction 100m"
        returns: array/tensor of eastern winds, with shape excluding last dimension
        """
        Vs = data[..., ::2]
        Vs = Vs * (1000 / 3600)  # km/h to m/s

        phis = np.deg2rad(data[..., 1::2])
     
        component = - Vs * trigeometry(phis)
        # legacy mean but also squeezes
        return component.mean(axis=-1)
    

class EastWindConverter(WindConverter):

    def om_to_era5(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return super().om_to_era5(data, trigeometry=np.sin)
    

class NorthWindConverter(WindConverter):

    def om_to_era5(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return super().om_to_era5(data, trigeometry=np.cos)
    

class SurfacePressureConverter(VariableConverter):
    def era5_to_om(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """pascal to hectopascal"""
        return data / 100

    def om_to_era5(self, data: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """hectopascal to pascal"""
        return data * 100
         

REGISTRY = {converter.data_var: converter for converter in [
        TemperatureConverter("2m_temperature", om_name="temperature_2m", short_code="t2m", unit="K"), 
        PrecipitationConverter("total_precipitation", om_name="precipitation", short_code="tp", unit="m/h"),
        EastWindConverter(
            "100m_u_component_of_wind", 
            om_name=["wind_speed_100m", "wind_direction_100m"],
            short_code="u100",
            unit="m/s",
        ),
        NorthWindConverter(
            "100m_v_component_of_wind", 
            om_name=["wind_speed_100m", "wind_direction_100m"],
            short_code="v100",
            unit="m/s",
        ),
        TemperatureConverter("2m_dewpoint_temperature", om_name="dew_point_2m", short_code="d2m", unit="K"),
        SurfacePressureConverter("surface_pressure", om_name="surface_pressure", short_code="sp", unit="Pa")
]}

def get_converter(data_var: str) -> VariableConverter:
    try:
        return REGISTRY[data_var]
    except KeyError:
        raise NotImplementedError(f"Variable {data_var} does not exist in registry")

