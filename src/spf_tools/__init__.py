"""
SPF Tools - Diagnostic and Utility Tools for See, Point, Fly Framework

This package contains various diagnostic and utility tools for the SPF framework,
organized into specific categories for different aspects of the system.

Modules:
    capture: Screen capture utilities with HiDPI/Retina display support
    diagnostics: System diagnostic tools for monitoring and troubleshooting
    vlm: VLM (Vision Language Model) testing and accuracy evaluation tools
    resolution: Monitor resolution management and scaling fixes
"""

__version__ = "0.1.0"
__author__ = "SPF Team"

# Import main diagnostic entry point
from .diagnostics.system_check import run_all_checks

# Import capture utilities
from .capture import capture_screen, capture_screen_resized, get_monitor_info

__all__ = [
    "run_all_checks",
    "capture_screen",
    "capture_screen_resized",
    "get_monitor_info"
]
