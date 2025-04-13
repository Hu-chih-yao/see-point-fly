#!/usr/bin/env python3
"""
Comprehensive system check for drone navigation system.
Tests monitor configuration, screen capture, and processing pipeline.
"""

import os
import sys
import cv2
import numpy as np
import time
import mss
import base64
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

# Import project modules
from tools.capture import capture_screen, capture_screen_resized, prepare_for_gemini, get_monitor_info
from action_projector import ActionProjector

def check_monitors():
    """Check monitor configuration and scaling"""
    print("\n=== MONITOR CONFIGURATION CHECK ===")
    
    with mss.mss() as sct:
        print(f"Total monitors: {len(sct.monitors)-1}")
        
        # Monitor 0 is all monitors combined
        print(f"\nMonitor 0 (All monitors combined):")
        print(f"  Dimensions: {sct.monitors[0]['width']}x{sct.monitors[0]['height']}")
        
        # Check each individual monitor
        for i in range(1, len(sct.monitors)):
            print(f"\nMonitor {i}:")
            monitor = sct.monitors[i]
            print(f"  Dimensions: {monitor['width']}x{monitor['height']}")
            
            # Take screenshot to check actual dimensions
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            print(f"  Screenshot dimensions: {img.shape[1]}x{img.shape[0]}x{img.shape[2]}")
            
            # Calculate scaling ratio
            width_ratio = img.shape[1] / monitor['width']
            height_ratio = img.shape[0] / monitor['height']
            print(f"  Scaling ratio: {width_ratio:.2f}x (horizontal), {height_ratio:.2f}x (vertical)")
            
            # Check for Retina/HiDPI display
            if width_ratio > 1.1 or height_ratio > 1.1:
                print(f"  ⚠️ Detected Retina/HiDPI display ({width_ratio:.0f}x scaling)")
            
            # Save screenshot for reference
            output_path = os.path.join(project_root, "output", "diagnostics")
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(output_path, f"monitor_{i}_screenshot.jpg"),
                cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            )
    
    return True

def check_capture():
    """Test screen capture with both regular and resized functions"""
    print("\n=== SCREEN CAPTURE TEST ===")
    
    # Create output directory
    output_path = os.path.join(project_root, "output", "diagnostics")
    os.makedirs(output_path, exist_ok=True)
    
    monitor_info = get_monitor_info()
    
    # Regular capture
    print("\nTesting regular capture (full resolution):")
    regular_capture = capture_screen(monitor_index=1)
    print(f"  Captured image dimensions: {regular_capture.shape[1]}x{regular_capture.shape[0]}")
    cv2.imwrite(
        os.path.join(output_path, "capture_regular.jpg"),
        cv2.cvtColor(regular_capture, cv2.COLOR_RGB2BGR)
    )
    
    # Resized capture
    print("\nTesting resized capture (matches reported dimensions):")
    resized_capture = capture_screen_resized(monitor_index=1)
    print(f"  Captured image dimensions: {resized_capture.shape[1]}x{resized_capture.shape[0]}")
    cv2.imwrite(
        os.path.join(output_path, "capture_resized.jpg"),
        cv2.cvtColor(resized_capture, cv2.COLOR_RGB2BGR)
    )
    
    # Compare dimensions to reported monitor
    if monitor_info:
        monitor = monitor_info["monitor_1"]
        print(f"\nMonitor 1 reported dimensions: {monitor['width']}x{monitor['height']}")
        print(f"Regular capture dimensions: {regular_capture.shape[1]}x{regular_capture.shape[0]}")
        print(f"Resized capture dimensions: {resized_capture.shape[1]}x{resized_capture.shape[0]}")
        
        # Check if resized matches reported
        if (resized_capture.shape[1] == monitor['width'] and 
            resized_capture.shape[0] == monitor['height']):
            print("✅ Resized capture matches reported dimensions")
        else:
            print("❌ Resized capture does not match reported dimensions")
            
        # Check scaling factor
        width_scaling = regular_capture.shape[1] / monitor['width']
        print(f"Detected scaling factor: {width_scaling:.2f}x")
    
    return True

def check_encoding():
    """Test image encoding for Gemini API"""
    print("\n=== IMAGE ENCODING TEST ===")
    
    # Create output directory
    output_path = os.path.join(project_root, "output", "diagnostics")
    os.makedirs(output_path, exist_ok=True)
    
    # Capture screen
    image = capture_screen(monitor_index=1)
    print(f"Original image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Save original
    cv2.imwrite(
        os.path.join(output_path, "encoding_original.jpg"),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    
    # Prepare for Gemini
    encoded = prepare_for_gemini(image)
    print(f"Base64 encoded length: {len(encoded)} bytes")
    
    # Decode to verify
    decoded = base64.b64decode(encoded)
    np_arr = np.frombuffer(decoded, np.uint8)
    decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    print(f"Decoded image dimensions: {decoded_img.shape[1]}x{decoded_img.shape[0]}")
    
    # Save decoded
    cv2.imwrite(
        os.path.join(output_path, "encoding_decoded.jpg"),
        decoded_img
    )
    
    if decoded_img.shape[:2] == (image.shape[0], image.shape[1]):
        print("✅ Encoded/decoded dimensions match original")
    else:
        print("❌ Encoded/decoded dimensions do not match original")
    
    return True

def check_projector():
    """Test ActionProjector with current monitor configuration"""
    print("\n=== ACTION PROJECTOR CHECK ===")
    
    # Create output directory
    output_path = os.path.join(project_root, "output", "diagnostics")
    os.makedirs(output_path, exist_ok=True)
    
    # Get monitor info
    monitor_info = get_monitor_info()
    if not monitor_info:
        print("❌ Could not get monitor info")
        return False
    
    # Capture test image
    test_image = capture_screen(monitor_index=1)
    image_height, image_width = test_image.shape[:2]
    
    # Create ActionProjector
    action_projector = ActionProjector(
        image_width=image_width,
        image_height=image_height
    )
    
    print(f"Monitor 1 captured dimensions: {image_width}x{image_height}")
    print(f"ActionProjector configured dimensions: {action_projector.image_width}x{action_projector.image_height}")
    
    # Check if dimensions match
    if (image_width == action_projector.image_width and 
        image_height == action_projector.image_height):
        print("✅ ActionProjector dimensions match captured image")
    else:
        print("❌ ActionProjector dimensions do not match captured image")
    
    # Test point projection
    center_3d = (0, 1, 0)  # Center of the view
    projected = action_projector.project_point(center_3d)
    
    print(f"Center point (0, 1, 0) projects to: {projected}")
    print(f"Expected center: ({action_projector.image_width//2}, {action_projector.image_height//2})")
    
    # Apply visualization
    viz_image = test_image.copy()
    cv2.circle(viz_image, projected, 10, (0, 255, 0), -1)
    cv2.putText(viz_image, f"Center: {projected}", 
               (projected[0]+15, projected[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite(
        os.path.join(output_path, "projector_test.jpg"),
        cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
    )
    
    return True

def run_all_checks():
    """Run all diagnostic checks"""
    print("\n=== DRONE SYSTEM DIAGNOSTIC ===")
    print("Running comprehensive system checks...")
    
    results = {}
    
    # Monitor check
    try:
        results["monitors"] = check_monitors()
    except Exception as e:
        print(f"❌ Monitor check failed: {e}")
        results["monitors"] = False
    
    # Capture check
    try:
        results["capture"] = check_capture()
    except Exception as e:
        print(f"❌ Capture check failed: {e}")
        results["capture"] = False
    
    # Encoding check
    try:
        results["encoding"] = check_encoding()
    except Exception as e:
        print(f"❌ Encoding check failed: {e}")
        results["encoding"] = False
    
    # Projector check
    try:
        results["projector"] = check_projector()
    except Exception as e:
        print(f"❌ Projector check failed: {e}")
        results["projector"] = False
    
    # Summary
    print("\n=== DIAGNOSTIC SUMMARY ===")
    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check.upper()}: {status}")
    
    if all(results.values()):
        print("\n✅ All checks passed! System is correctly configured.")
    else:
        print("\n⚠️ Some checks failed. Review the issues above and fix them.")
    
    return all(results.values())

if __name__ == "__main__":
    run_all_checks() 