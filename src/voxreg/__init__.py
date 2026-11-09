"""Physical-space image registration using xarray-based VoxelData."""

from voxreg.geometry import position_to_world
from voxreg.resampling import resample
from voxreg.rigid import RegistrationResult, register
from voxreg.stages import Stage

__all__ = [
    "RegistrationResult",
    "Stage",
    "position_to_world",
    "register",
    "resample",
]
