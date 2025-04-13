"""
Centralized screen capture utilities.
Handles different scaling factors and monitor configurations.
"""

import os
import cv2
import numpy as np
import mss
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

def capture_screen(monitor_index=1):
    """
    Capture screen with automatic scaling detection and adjustment.
    
    Args:
        monitor_index (int): Monitor to capture (1=primary, 2=secondary, etc.)
        
    Returns:
        np.ndarray: RGB image with proper dimensions
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
            
            # If HiDPI/Retina display detected
            if width_ratio > 1.1 or height_ratio > 1.1:
                print(f"HiDPI display detected (scaling: {width_ratio:.2f}x)")
                # For ActionProjector, we want to keep the full resolution
                return img_rgb
            else:
                return img_rgb
    except Exception as e:
        print(f"Error capturing screen: {e}")
        # Return blank image with error message
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(blank, f"Screen capture error: {e}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return blank

def capture_screen_resized(monitor_index=1):
    """
    Capture screen and resize to match reported monitor dimensions.
    Use this when you need images that match the logical resolution.
    
    Args:
        monitor_index (int): Monitor to capture
        
    Returns:
        np.ndarray: RGB image resized to match reported dimensions
    """
    with mss.mss() as sct:
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

def prepare_for_gemini(image_rgb):
    """
    Prepare an RGB image for sending to Gemini API.
    Converts RGB to BGR for proper OpenCV encoding.
    
    Args:
        image_rgb (np.ndarray): RGB image to prepare
        
    Returns:
        str: Base64-encoded image string ready for Gemini
    """
    import base64
    
    # Convert RGB to BGR for proper encoding
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', image_bgr)
    
    # Convert to base64 string
    return base64.b64encode(buffer).decode('utf-8')

def get_monitor_info():
    """
    Get information about all available monitors.
    
    Returns:
        dict: Information about monitors including dimensions and scaling
    """
    with mss.mss() as sct:
        info = {}
        
        for i, monitor in enumerate(sct.monitors):
            # Capture a small portion to detect scaling
            if i > 0:  # Skip monitor 0 (all monitors combined)
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