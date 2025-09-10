"""
Diagnostic Tools for SPF Framework

This module provides comprehensive diagnostic tools for checking system health,
monitor configuration, screen capture functionality, and VLM client operations.

Functions:
    run_all_checks: Execute all system diagnostic checks
    check_monitors: Verify monitor configuration and scaling
    check_capture: Test screen capture functionality
    check_vlm_client: Test VLM client initialization and basic operations
    check_projector: Test ActionProjector functionality
    check_encoding: Test image encoding pipeline for VLM APIs
"""

from .system_check import (
    run_all_checks,
    check_monitors,
    check_capture,
    check_vlm_client,
    check_projector,
    check_encoding
)

__all__ = [
    "run_all_checks",
    "check_monitors",
    "check_capture",
    "check_vlm_client",
    "check_projector",
    "check_encoding"
]
