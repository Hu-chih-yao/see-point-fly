"""
Screen Capture Utilities for SPF Framework

This module provides centralized screen capture utilities that handle different
scaling factors and monitor configurations, with proper support for HiDPI/Retina displays.

Functions:
    capture_screen: Capture screen with automatic scaling detection
    capture_screen_resized: Capture and resize to match logical resolution
    prepare_for_vlm: Prepare images for VLM API transmission
    get_monitor_info: Get detailed monitor configuration information
"""

import os
import cv2
import numpy as np
import mss
import base64
from typing import Dict, Tuple, Optional


def capture_screen(monitor_index: int = 1) -> np.ndarray:
    """
    Capture screen with automatic scaling detection and adjustment.

    Args:
        monitor_index: Monitor to capture (1=primary, 2=secondary, etc.)

    Returns:
        RGB image array with proper dimensions

    Raises:
        RuntimeError: If screen capture fails
    """
    try:
        with mss.mss() as sct:
            if monitor_index >= len(sct.monitors):
                monitor_index = 1

            monitor = sct.monitors[monitor_index]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            # Calculate scaling ratio
            width_ratio = img_rgb.shape[1] / monitor['width']
            height_ratio = img_rgb.shape[0] / monitor['height']

            # Log HiDPI detection for debugging
            if width_ratio > 1.1 or height_ratio > 1.1:
                print(f"HiDPI display detected (scaling: {width_ratio:.2f}x)")

            return img_rgb

    except Exception as e:
        raise RuntimeError(f"Screen capture failed: {e}")


def capture_screen_resized(monitor_index: int = 1) -> np.ndarray:
    """
    Capture screen and resize to match reported monitor dimensions.
    Use this when you need images that match the logical resolution.

    Args:
        monitor_index: Monitor to capture

    Returns:
        RGB image array resized to match reported dimensions

    Raises:
        RuntimeError: If screen capture fails
    """
    try:
        with mss.mss() as sct:
            if monitor_index >= len(sct.monitors):
                monitor_index = 1

            monitor = sct.monitors[monitor_index]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            # Calculate scaling ratio
            width_ratio = img_rgb.shape[1] / monitor['width']
            height_ratio = img_rgb.shape[0] / monitor['height']

            # If Retina/HiDPI display detected, resize to match reported dimensions
            if width_ratio > 1.1 or height_ratio > 1.1:
                resized_img = cv2.resize(
                    img_rgb,
                    (monitor['width'], monitor['height']),
                    interpolation=cv2.INTER_AREA
                )
                return resized_img
            else:
                return img_rgb

    except Exception as e:
        raise RuntimeError(f"Screen capture failed: {e}")


def prepare_for_vlm(image_rgb: np.ndarray) -> str:
    """
    Prepare an RGB image for VLM API transmission.
    Converts RGB to BGR for proper OpenCV encoding and returns base64 string.

    Args:
        image_rgb: RGB image array to prepare

    Returns:
        Base64-encoded image string ready for VLM APIs

    Raises:
        ValueError: If image encoding fails
    """
    try:
        # Convert RGB to BGR for proper encoding
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', image_bgr)
        if not success:
            raise ValueError("Failed to encode image to JPEG")

        # Convert to base64 string
        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        raise ValueError(f"Image preparation failed: {e}")


def get_monitor_info() -> Dict[str, Dict]:
    """
    Get detailed information about all available monitors.

    Returns:
        Dictionary containing monitor information including dimensions and scaling

    Example:
        {
            "monitor_1": {
                "index": 1,
                "width": 1920,
                "height": 1080,
                "captured_width": 3840,
                "captured_height": 2160,
                "width_scaling": 2.0,
                "height_scaling": 2.0,
                "is_hidpi": True
            }
        }
    """
    try:
        with mss.mss() as sct:
            info = {}

            for i, monitor in enumerate(sct.monitors):
                if i > 0:  # Skip monitor 0 (all monitors combined)
                    # Capture a small portion to detect scaling
                    screenshot = sct.grab(monitor)
                    img = np.array(screenshot)

                    # Calculate scaling
                    width_ratio = img.shape[1] / monitor['width']
                    height_ratio = img.shape[0] / monitor['height']

                    info[f"monitor_{i}"] = {
                        "index": i,
                        "width": monitor['width'],
                        "height": monitor['height'],
                        "captured_width": img.shape[1],
                        "captured_height": img.shape[0],
                        "width_scaling": width_ratio,
                        "height_scaling": height_ratio,
                        "is_hidpi": width_ratio > 1.1 or height_ratio > 1.1
                    }

            return info

    except Exception as e:
        print(f"Warning: Could not get monitor info: {e}")
        return {}


def create_test_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    Create a test image with grid pattern for diagnostic purposes.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        RGB test image with grid pattern and coordinate markers
    """
    # Create blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add grid pattern
    grid_spacing = 100
    for i in range(0, width, grid_spacing):
        cv2.line(image, (i, 0), (i, height), (64, 64, 64), 1)
    for i in range(0, height, grid_spacing):
        cv2.line(image, (0, i), (width, i), (64, 64, 64), 1)

    # Add center crosshair
    center_x, center_y = width // 2, height // 2
    cv2.line(image, (center_x - 50, center_y), (center_x + 50, center_y), (0, 255, 0), 3)
    cv2.line(image, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 0), 3)

    # Add corner markers
    corner_size = 20
    # Top-left
    cv2.rectangle(image, (0, 0), (corner_size, corner_size), (255, 0, 0), -1)
    # Top-right
    cv2.rectangle(image, (width - corner_size, 0), (width, corner_size), (0, 255, 0), -1)
    # Bottom-left
    cv2.rectangle(image, (0, height - corner_size), (corner_size, height), (0, 0, 255), -1)
    # Bottom-right
    cv2.rectangle(image, (width - corner_size, height - corner_size), (width, height), (255, 255, 0), -1)

    # Add dimension text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"{width}x{height}", (10, 30), font, 1, (255, 255, 255), 2)

    return image


__all__ = [
    "capture_screen",
    "capture_screen_resized",
    "prepare_for_vlm",
    "get_monitor_info",
    "create_test_image"
]
