#!/usr/bin/env python3
"""
Resolution fixing tool to diagnose and fix dimension mismatches
This helps identify why screen captures have different dimensions than monitor settings
"""

import cv2
import numpy as np
import mss
import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from action_projector import ActionProjector
    from drone_controller import capture_screen, print_monitor_info
except ImportError:
    print("Error importing project modules. Make sure your PYTHONPATH includes the drone project root directory.")
    sys.exit(1)

# Directory for saving test images
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/screenshots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Directory for saving the fixed capture script
TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../capture'))
os.makedirs(TOOLS_DIR, exist_ok=True)

def detailed_monitor_check():
    """Perform detailed check of monitors and pixel ratios"""
    print("\n=== DETAILED MONITOR ANALYSIS ===")
    
    with mss.mss() as sct:
        # Analyze all monitors
        for i in range(len(sct.monitors)):
            monitor = sct.monitors[i]
            print(f"\nMonitor {i}:")
            print(f"  Dimensions: {monitor['width']}x{monitor['height']}")
            
            if i > 0:  # Skip the "all monitors" virtual screen
                # Take a screenshot and analyze dimensions
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                print(f"  Screenshot dimensions: {img.shape[1]}x{img.shape[0]}x{img.shape[2]}")
                
                # Calculate monitor scaling ratio
                width_ratio = img.shape[1] / monitor['width']
                height_ratio = img.shape[0] / monitor['height']
                print(f"  Scaling ratio: {width_ratio:.2f}x (horizontal), {height_ratio:.2f}x (vertical)")
                
                if abs(width_ratio - 2.0) < 0.1 or abs(height_ratio - 2.0) < 0.1:
                    print("  ⚠️ Detected Retina/HiDPI display (2x scaling)")
                elif abs(width_ratio - 1.5) < 0.1 or abs(height_ratio - 1.5) < 0.1:
                    print("  ⚠️ Detected 1.5x display scaling")
                
                # Save a sample screenshot for analysis
                cv2.imwrite(f"monitor_{i}_screenshot.jpg", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
                print(f"  Saved sample to monitor_{i}_screenshot.jpg")

def create_resolution_fix():
    """Create fixed capture functions that account for monitor scaling"""
    print("\n=== CREATING RESOLUTION FIX ===")
    
    # Calculate primary monitor scaling
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        
        width_ratio = img.shape[1] / monitor['width']
        height_ratio = img.shape[0] / monitor['height']
        
        print(f"Monitor 1 has {width_ratio:.2f}x scaling")
        
        # Determine if we need the fix
        if abs(width_ratio - 1.0) < 0.1:
            print("No scaling detected, fix not needed.")
            return
        
        # Create the capture_screen_fixed.py file
        fix_code = f"""#!/usr/bin/env python3
'''
Resolution-fixed screen capture function
Generated to handle {width_ratio:.2f}x display scaling
'''

import cv2
import numpy as np
import mss

def capture_screen_fixed(monitor_index=1):
    '''
    Capture the screen with resolution correction for HiDPI/Retina displays
    Returns image with dimensions matching reported monitor dimensions, not actual pixels
    '''
    try:
        with mss.mss() as sct:
            # Get monitor information
            if monitor_index >= len(sct.monitors):
                print(f"Warning: Monitor index {{monitor_index}} out of range. Using main monitor (1).")
                monitor_index = 1
                
            monitor = sct.monitors[monitor_index]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Get the actual dimensions
            actual_height, actual_width = img_rgb.shape[:2]
            
            # Calculate scaling ratio
            width_ratio = actual_width / monitor['width']
            height_ratio = actual_height / monitor['height']
            
            # If we detect HiDPI/Retina scaling, resize the image to match reported dimensions
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
        print(f"Error capturing screen: {{e}}")
        # Return a blank image with error message as fallback
        blank = np.zeros((monitor['height'], monitor['width'], 3), dtype=np.uint8)
        cv2.putText(blank, "Screen capture error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return blank

def capture_screen_raw(monitor_index=1):
    '''
    Capture the screen without any resolution correction
    Returns the raw captured image with actual pixel dimensions
    '''
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[monitor_index]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    except Exception as e:
        print(f"Error capturing screen: {{e}}")
        return None

if __name__ == "__main__":
    # Test the function
    print("Testing resolution-fixed screen capture...")
    
    # Capture with both functions
    start_time = time.time()
    img_fixed = capture_screen_fixed(1)
    img_raw = capture_screen_raw(1)
    
    print(f"Fixed image dimensions: {{img_fixed.shape[1]}}x{{img_fixed.shape[0]}}")
    if img_raw is not None:
        print(f"Raw image dimensions: {{img_raw.shape[1]}}x{{img_raw.shape[0]}}")
    
    # Save for comparison
    cv2.imwrite("capture_fixed.jpg", cv2.cvtColor(img_fixed, cv2.COLOR_RGB2BGR))
    if img_raw is not None:
        cv2.imwrite("capture_raw.jpg", cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR))
        
    print(f"Capture completed in {{time.time() - start_time:.2f}} seconds")
    print("Images saved to capture_fixed.jpg and capture_raw.jpg")
"""
        
        # Write the file
        with open("capture_screen_fixed.py", "w") as f:
            f.write(fix_code)
        
        print(f"Created capture_screen_fixed.py with {width_ratio:.2f}x scaling correction")
        print("This file provides two functions:")
        print("1. capture_screen_fixed() - Returns image matching monitor reported dimensions")
        print("2. capture_screen_raw() - Returns image with actual pixel dimensions")
        print("\nTo use this fix:")
        print("1. Import these functions instead of the original capture_screen:")
        print("   from capture_screen_fixed import capture_screen_fixed as capture_screen")
        print("2. Or update ActionProjector to match actual pixel dimensions:")
        print(f"   self.image_width = {img.shape[1]}  # Raw pixel count")
        print(f"   self.image_height = {img.shape[0]} # Raw pixel count")

def test_capture_functions():
    """Test different capture approaches and compare"""
    print("\n=== TESTING CAPTURE METHODS ===")
    
    methods = [
        ("Standard MSS", "Uses standard mss library approach"),
        ("OpenCV alternative", "Uses cv2.VideoCapture approach")
    ]
    
    # Method 1: Standard MSS
    try:
        print(f"\n1. {methods[0][0]}: {methods[0][1]}")
        start_time = time.time()
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        print(f"  Capture time: {time.time() - start_time:.3f} seconds")
        print(f"  Image dimensions: {rgb_img.shape[1]}x{rgb_img.shape[0]}")
        cv2.imwrite("capture_mss.jpg", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        print("  Saved to capture_mss.jpg")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Method 2: OpenCV approach
    try:
        print(f"\n2. {methods[1][0]}: {methods[1][1]}")
        start_time = time.time()
        
        # Try using OpenCV's VideoCapture for screen capture
        # Note: This might not work on all platforms
        cap = cv2.VideoCapture(0)  # Try default camera first
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"  Capture time: {time.time() - start_time:.3f} seconds")
            print(f"  Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imwrite("capture_opencv.jpg", frame)
            print("  Saved to capture_opencv.jpg")
        else:
            print("  Failed to capture using OpenCV")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nRecommendation:")
    print("Continue using mss for screen capture, but adapt your ActionProjector")
    print("to either resize captured images or update its dimensions accordingly.")

if __name__ == "__main__":
    print("=== RESOLUTION FIX TOOL ===")
    print("Analyzing monitor setup and creating resolution fixes...")
    
    detailed_monitor_check()
    create_resolution_fix()
    
    print("\nAnalysis and fix generation complete.")
    print("To test the fix, run: python capture_screen_fixed.py") 