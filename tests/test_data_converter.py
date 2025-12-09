"""
Unit tests for zeus.data.era5.converter module.
"""
import pytest
import warnings
import numpy as np
import torch
from zeus.data.era5.converter import (
    VariableConverter,
    TemperatureConverter,
    PrecipitationConverter,
    WindConverter,
    EastWindConverter,
    NorthWindConverter,
    SurfacePressureConverter,
    get_converter,
    REGISTRY,
)


class TestTemperatureConverter:
    """Tests for TemperatureConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a TemperatureConverter instance."""
        return TemperatureConverter(
            data_var="2m_temperature",
            om_name="temperature_2m",
            short_code="t2m",
            unit="K"
        )

    def test_era5_to_om_float(self, converter):
        """Test converting ERA5 to OpenMeteo for float."""
        # ERA5 is in Kelvin, OpenMeteo is in Celsius
        result = converter.era5_to_om(273.15)
        assert abs(result - 0.0) < 1e-6

    def test_era5_to_om_array(self, converter):
        """Test converting ERA5 to OpenMeteo for numpy array."""
        era5_data = np.array([273.15, 283.15, 293.15])
        result = converter.era5_to_om(era5_data)
        expected = np.array([0.0, 10.0, 20.0])
        assert np.allclose(result, expected)

    def test_era5_to_om_tensor(self, converter):
        """Test converting ERA5 to OpenMeteo for torch tensor."""
        era5_data = torch.tensor([273.15, 283.15, 293.15])
        result = converter.era5_to_om(era5_data)
        expected = torch.tensor([0.0, 10.0, 20.0])
        assert torch.allclose(result, expected)

    def test_om_to_era5_float(self, converter):
        """Test converting OpenMeteo to ERA5 for float."""
        # OpenMeteo is in Celsius, ERA5 is in Kelvin
        result = converter.om_to_era5(0.0)
        assert abs(result - 273.15) < 1e-6

    def test_om_to_era5_array(self, converter):
        """Test converting OpenMeteo to ERA5 for numpy array."""
        om_data = np.array([0.0, 10.0, 20.0])
        result = converter.om_to_era5(om_data)
        expected = np.array([273.15, 283.15, 293.15])
        assert np.allclose(result, expected)

    def test_om_to_era5_tensor(self, converter):
        """Test converting OpenMeteo to ERA5 for torch tensor."""
        om_data = torch.tensor([0.0, 10.0, 20.0])
        result = converter.om_to_era5(om_data)
        expected = torch.tensor([273.15, 283.15, 293.15])
        assert torch.allclose(result, expected)

    def test_roundtrip_conversion(self, converter):
        """Test that conversion is reversible."""
        original = 300.0
        converted = converter.era5_to_om(original)
        back = converter.om_to_era5(converted)
        assert abs(original - back) < 1e-6


class TestPrecipitationConverter:
    """Tests for PrecipitationConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a PrecipitationConverter instance."""
        return PrecipitationConverter(
            data_var="total_precipitation",
            om_name="precipitation",
            short_code="tp",
            unit="m/h"
        )

    def test_era5_to_om_float(self, converter):
        """Test converting ERA5 to OpenMeteo for float."""
        # ERA5 is in m/h, OpenMeteo is in mm/h
        result = converter.era5_to_om(0.001)
        assert abs(result - 1.0) < 1e-6

    def test_om_to_era5_float(self, converter):
        """Test converting OpenMeteo to ERA5 for float."""
        # OpenMeteo is in mm/h, ERA5 is in m/h
        result = converter.om_to_era5(1.0)
        assert abs(result - 0.001) < 1e-6

    def test_roundtrip_conversion(self, converter):
        """Test that conversion is reversible."""
        original = 0.005
        converted = converter.era5_to_om(original)
        back = converter.om_to_era5(converted)
        assert abs(original - back) < 1e-9


class TestEastWindConverter:
    """Tests for EastWindConverter class."""

    @pytest.fixture
    def converter(self):
        """Create an EastWindConverter instance."""
        return EastWindConverter(
            data_var="100m_u_component_of_wind",
            om_name=["wind_speed_100m", "wind_direction_100m"],
            short_code="u100",
            unit="m/s"
        )

    def test_om_to_era5_numpy(self, converter):
        """Test converting OpenMeteo wind to ERA5 east wind component with numpy array."""
        # Create test data: shape [time, lat, lon, variables]
        # For a single time, 2 lat, 2 lon with 2 variables (wind_speed, wind_direction), use shape [1, 2, 2, 2]
        # Wind speed: 36 km/h = 10 m/s, Direction: 0 degrees (from north)
        om_data = np.array([[[[36.0, 0.0], [36.0, 0.0]], [[36.0, 0.0], [36.0, 0.0]]]], dtype=np.float32)
        result = converter.om_to_era5(om_data)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
        # Should have shape [1, 2, 2] - batch dimensions preserved after mean over last dimension
        assert result.shape == (1, 2, 2)
        # Wind from north (0 deg) -> zero east component
        # sin(0) = 0, so component should be 0
        assert abs(result[0, 0, 0] - 0.0) < 1e-6

    def test_om_to_era5_torch(self, converter):
        """Test converting OpenMeteo wind to ERA5 with torch tensor."""
        # Shape [1, 2, 2, 2] - single time, 2 lat, 2 lon with 2 variables (wind_speed, wind_direction)
        om_data = torch.tensor([[[[36.0, 90.0], [36.0, 90.0]], [[36.0, 90.0], [36.0, 90.0]]]], dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*__array_wrap__.*", category=DeprecationWarning)
            result = converter.om_to_era5(om_data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 2, 2)
        # Wind from east (90 deg) -> negative east component
        # sin(90) = 1, so component should be -10
        assert abs(result[0, 0, 0].item() - (-10.0)) < 1e-6


class TestNorthWindConverter:
    """Tests for NorthWindConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a NorthWindConverter instance."""
        return NorthWindConverter(
            data_var="100m_v_component_of_wind",
            om_name=["wind_speed_100m", "wind_direction_100m"],
            short_code="v100",
            unit="m/s"
        )

    def test_om_to_era5_numpy(self, converter):
        """Test converting OpenMeteo wind to ERA5 north wind component with numpy array."""
        # Create test data: shape [time, lat, lon, variables]
        # For a single time, 2 lat, 2 lon with 2 variables (wind_speed, wind_direction), use shape [1, 2, 2, 2]
        # Wind from north (0 degrees) should give negative north component
        # Wind speed: 36 km/h = 10 m/s
        om_data = np.array([[[[36.0, 0.0], [36.0, 0.0]], [[36.0, 0.0], [36.0, 0.0]]]], dtype=np.float32)
        result = converter.om_to_era5(om_data)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
        # Should have shape [1, 2, 2] - batch dimensions preserved after mean over last dimension
        assert result.shape == (1, 2, 2)
        # Wind from north (0 deg) -> negative north component
        # cos(0) = 1, so component should be -10 (negative because of the minus sign in formula)
        assert abs(result[0, 0, 0] - (-10.0)) < 1e-6

    def test_om_to_era5_torch(self, converter):
        """Test converting OpenMeteo wind to ERA5 north wind component with torch tensor."""
        # Create test data: shape [time, lat, lon, variables]
        # For a single time, 2 lat, 2 lon with 2 variables (wind_speed, wind_direction), use shape [1, 2, 2, 2]
        # Wind from north (0 degrees) should give negative north component
        # Wind speed: 36 km/h = 10 m/s
        om_data = torch.tensor([[[[36.0, 0.0], [36.0, 0.0]], [[36.0, 0.0], [36.0, 0.0]]]], dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*__array_wrap__.*", category=DeprecationWarning)
            result = converter.om_to_era5(om_data)
        
        # Should return torch tensor
        assert isinstance(result, torch.Tensor)
        # Should have shape [1, 2, 2] - batch dimensions preserved after mean over last dimension
        assert result.shape == (1, 2, 2)
        # Wind from north (0 deg) -> negative north component
        # cos(0) = 1, so component should be -10 (negative because of the minus sign in formula)
        assert abs(result[0, 0, 0].item() - (-10.0)) < 1e-6


class TestSurfacePressureConverter:
    """Tests for SurfacePressureConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a SurfacePressureConverter instance."""
        return SurfacePressureConverter(
            data_var="surface_pressure",
            om_name="surface_pressure",
            short_code="sp",
            unit="Pa"
        )

    def test_era5_to_om_float(self, converter):
        """Test converting ERA5 to OpenMeteo for float."""
        # ERA5 is in Pascal, OpenMeteo is in hectopascal
        result = converter.era5_to_om(101325.0)
        assert abs(result - 1013.25) < 1e-6

    def test_om_to_era5_float(self, converter):
        """Test converting OpenMeteo to ERA5 for float."""
        # OpenMeteo is in hectopascal, ERA5 is in Pascal
        result = converter.om_to_era5(1013.25)
        assert abs(result - 101325.0) < 1e-6

    def test_roundtrip_conversion(self, converter):
        """Test that conversion is reversible."""
        original = 101325.0
        converted = converter.era5_to_om(original)
        back = converter.om_to_era5(converted)
        assert abs(original - back) < 1e-6


class TestGetConverter:
    """Tests for get_converter function."""

    def test_get_converter_existing(self):
        """Test getting an existing converter."""
        converter = get_converter("2m_temperature")
        assert isinstance(converter, TemperatureConverter)
        assert converter.data_var == "2m_temperature"

    def test_get_converter_all_registry_vars(self):
        """Test that all variables in registry can be retrieved."""
        for var_name in REGISTRY.keys():
            converter = get_converter(var_name)
            assert converter.data_var == var_name

    def test_get_converter_nonexistent(self):
        """Test that getting a nonexistent converter raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            get_converter("nonexistent_variable")
        assert "nonexistent_variable" in str(exc_info.value)

