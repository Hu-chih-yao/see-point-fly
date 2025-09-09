"""
Projectors module for SPF (See, Point, Fly)

This module contains action projectors that handle coordinate transformations
and spatial reasoning for different platforms:
- ActionProjector: For real camera systems
- ActionProjectorSim: For simulator environments
"""

from .action_projector import ActionProjector
from .action_projector_sim import ActionProjectorSim

__all__ = [
    "ActionProjector",
    "ActionProjectorSim"
]
