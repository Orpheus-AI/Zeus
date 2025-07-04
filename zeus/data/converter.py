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
        "data: [batch..., 4]",
        "return: [batch...]",
    )
    def om_to_era5(self, data: Union[np.ndarray, torch.Tensor], trigeometry: Callable) -> Union[np.ndarray, torch.Tensor]:
        """
        OpenMeteo only provides overall wind speed (km/h) and wind direction at 80 or 120 meters.
        So we convert this to eastern wind at a 100 meters (m/s). We average altitudes.
        See: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398

        data: Any array/tensor with last dimension being the variables (in order!):
            "wind speed 80m", "wind direction 80m", "wind speed 120m", "wind direction 120m"
        returns: array/tensor of eastern winds, with shape excluding last dimension
        """
        Vs = data[..., torch.arange(0, data.shape[-1], 2)]
        Vs *= 1000 / 3600 # km/h to m/s

        phis = np.deg2rad(data[..., torch.arange(1, data.shape[-1], 2)])
     
        component = - Vs * trigeometry(phis)
        # take average of 80 and 120 meters for 100m
        return component.mean(axis=-1)
    

class EastWindConverter(WindConverter):

    def om_to_era5(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return super().om_to_era5(data, trigeometry=np.sin)
    

class NorthWindConverter(WindConverter):

    def om_to_era5(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return super().om_to_era5(data, trigeometry=np.cos)
     

REGISTRY = {converter.data_var: converter for converter in [
        TemperatureConverter("2m_temperature", om_name="temperature_2m", short_code="t2m", unit="K"), 
        PrecipitationConverter("total_precipitation", om_name="precipitation", short_code="tp", unit="m/h"),
        EastWindConverter(
            "100m_u_component_of_wind", 
            om_name=["wind_speed_80m", "wind_direction_80m", "wind_speed_120m", "wind_direction_120m"],
            short_code="u100",
            unit="m/s",
        ),
        NorthWindConverter(
            "100m_v_component_of_wind", 
            om_name=["wind_speed_80m", "wind_direction_80m", "wind_speed_120m", "wind_direction_120m"],
            short_code="v100",
            unit="m/s",
        )
]}

def get_converter(data_var: str) -> VariableConverter:
    try:
        return REGISTRY[data_var]
    except KeyError:
        raise NotImplementedError(f"Variable {data_var} does not exist in registry")

