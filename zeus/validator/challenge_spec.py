from dataclasses import dataclass
from typing import Dict, Tuple

from zeus.base.dendrite import DendriteSettings


def make_state_key(variable: str, start_offset: int, end_offset: int) -> str:
    return f"{variable}@{start_offset}_{end_offset}"


@dataclass(frozen=True)
class ChallengeSpec:
    variable: str
    start_offset: int
    end_offset: int
    weight: float
    prediction_dendrite_settings: DendriteSettings

    @property
    def state_key(self) -> str:
        return make_state_key(self.variable, self.start_offset, self.end_offset)


def build_challenge_registry(
    era5_data_vars: Dict[str, float],
    time_windows: list[Tuple[int, int]],
    prediction_settings_per_window: Dict[Tuple[int, int], DendriteSettings],
) -> Dict[str, ChallengeSpec]:
    """Build {state_key: ChallengeSpec} from ERA5 variables × time windows.

    Each variable's total weight is split equally across its windows.
    prediction_settings_per_window maps (start_offset, end_offset) -> DendriteSettings.
    """
    n_windows = len(time_windows)
    registry: Dict[str, ChallengeSpec] = {}
    for variable, total_weight in era5_data_vars.items():
        per_window_weight = total_weight / n_windows
        for start_offset, end_offset in time_windows:
            settings = prediction_settings_per_window[(start_offset, end_offset)]
            spec = ChallengeSpec(
                variable=variable,
                start_offset=start_offset,
                end_offset=end_offset,
                weight=per_window_weight,
                prediction_dendrite_settings=settings,
            )
            registry[spec.state_key] = spec
    return registry


def offsets_from_predict_hours(
    predict_hours: int,
    time_windows: list[Tuple[int, int]],
    step_size: int = 1,
) -> Tuple[int, int]:
    """Derive (start_offset, end_offset) from predict_hours by matching against known windows."""
    for start_h, end_h in time_windows:
        expected = (end_h - start_h) // step_size + 1
        if predict_hours == expected:
            return start_h, end_h
    raise ValueError(
        f"Cannot derive offsets from predict_hours={predict_hours}, "
        f"known windows={time_windows}"
    )
