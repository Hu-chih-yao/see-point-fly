"""
SPF (See, Point, Fly) - A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation
"""

__version__ = "0.1.0"
__author__ = "SPF Team"
__description__ = "See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation"

# Import main classes for easy access
from .clients.vlm_client import VLMClient
from .spaces.drone_space import DroneActionSpace, ActionPoint

__all__ = [
    "VLMClient",
    "DroneActionSpace",
    "ActionPoint"
]
