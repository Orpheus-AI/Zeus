from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from check_shapes import check_shapes

@dataclass
class WeatherXMVariable(ABC):
    """
    Utilities to convert WeatherXM's va
    and vice versa
    """
    data_var: str
    unit: str

REGISTRY = {converter.data_var: converter for converter in [
    WeatherXMVariable("temperature", unit="°C"),
    WeatherXMVariable("wind_speed", unit="m/s"),
    WeatherXMVariable("wind_direction", unit="°"),
    WeatherXMVariable("humidity", unit="%"),
    WeatherXMVariable("pressure", unit="hPa"),
]}

def get_wxm_variable(variable: str) -> WeatherXMVariable:
    try:
        return REGISTRY[variable]
    except KeyError:
        raise NotImplementedError(f"Variable {variable} does not exist in registry")