#!/usr/bin/env python3
"""
Resolution verification tool for drone navigation system
Checks monitor resolutions and traces image dimensions throughout processing
"""

import cv2
import numpy as np
import mss
import time
import sys
import os
import json
import base64
from pathlib import Path

# Add project root to path to allow importing from root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import project modules - add error handling in case paths aren't set up
try:
    from action_projector import ActionProjector
    from VLM_Tello_integration.drone_controller_sim import DroneController, capture_screen, print_monitor_info
except ImportError:
    print("Error importing project modules. Make sure your PYTHONPATH includes the drone project root directory.")
    sys.exit(1)

def check_monitor_resolutions():
    """Check and report all monitor resolutions"""
    print("\n=== MONITOR RESOLUTIONS ===")
    
    with mss.mss() as sct:
        print(f"Total monitors: {len(sct.monitors)}")
        
        # Monitor 0 is special - it's the "all monitors" virtual screen
        print(f"\nMonitor 0 (All monitors combined):")
        m0 = sct.monitors[0]
        print(f"  Dimensions: {m0['width']}x{m0['height']}")
        print(f"  Position: Left={m0['left']}, Top={m0['top']}")
        
        # Print individual monitor details
        for i in range(1, len(sct.monitors)):
            monitor = sct.monitors[i]
            print(f"\nMonitor {i}:")
            print(f"  Dimensions: {monitor['width']}x{monitor['height']}")
            print(f"  Position: Left={monitor['left']}, Top={monitor['top']}, "
                  f"Right={monitor['left']+monitor['width']}, Bottom={monitor['top']+monitor['height']}")
    
    # Check what's configured in ActionProjector
    try:
        ap = ActionProjector()
        print(f"\nActionProjector configured dimensions: {ap.image_width}x{ap.image_height}")
        
        # Verify if dimensions match any monitor
        with mss.mss() as sct:
            match_found = False
            for i in range(1, len(sct.monitors)):
                monitor = sct.monitors[i]
                if monitor['width'] == ap.image_width and monitor['height'] == ap.image_height:
                    print(f"✅ MATCH: ActionProjector dimensions match Monitor {i}")
                    match_found = True
            
            if not match_found:
                print("❌ ERROR: ActionProjector dimensions don't match any monitor")
                
                # Recommend fixing this
                print("\nRecommendation: Update ActionProjector.__init__ with your actual monitor resolution:")
                
                # Suggest code to use Monitor 1 by default
                m1 = sct.monitors[1]
                print(f"""
    def __init__(self):
        # Camera parameters for proper scaling
        self.image_width = {m1['width']}   # Updated to match current monitor resolution
        self.image_height = {m1['height']}  # Updated to match current monitor resolution
                """)
    except Exception as e:
        print(f"Error checking ActionProjector configuration: {e}")

def check_processing_pipeline():
    """Trace image dimensions throughout the processing pipeline"""
    print("\n=== IMAGE PROCESSING PIPELINE CHECK ===")
    
    try:
        # Phase 1: Screen Capture
        print("\nPhase 1: Screen Capture")
        start_time = time.time()
        with mss.mss() as sct:
            # Try both monitor 0 and 1
            for monitor_idx in [1, 0]:
                try:
                    monitor = sct.monitors[monitor_idx]
                    screenshot = sct.grab(monitor)
                    img = np.array(screenshot)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    
                    print(f"  Monitor {monitor_idx} screen capture:")
                    print(f"    - Raw screenshot dimensions: {img.shape[1]}x{img.shape[0]}x{img.shape[2]}")
                    print(f"    - After RGB conversion: {rgb_img.shape[1]}x{rgb_img.shape[0]}x{rgb_img.shape[2]}")
                    
                    # Save diagnostic image
                    cv2.imwrite(f"captured_monitor_{monitor_idx}.jpg", 
                               cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                    print(f"    - Saved to captured_monitor_{monitor_idx}.jpg")
                except Exception as e:
                    print(f"    - Error capturing monitor {monitor_idx}: {e}")
        
        # Phase 2: ActionProjector
        print("\nPhase 2: ActionProjector Processing")
        ap = ActionProjector()
        
        # Check dimensions configured in ActionProjector
        print(f"  ActionProjector configured for: {ap.image_width}x{ap.image_height}")
        
        # Load one of our captures for testing
        test_image = cv2.imread("captured_monitor_1.jpg")
        if test_image is None:
            # If we couldn't load the captured image, try the test image
            test_image = cv2.imread("frame_1733321874.11946.jpg")
            print("  Using sample test image instead of capture")
        
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"  Test image dimensions: {test_image.shape[1]}x{test_image.shape[0]}")
        
        # Phase 3: Check 3D point projection
        print("\nPhase 3: Testing 3D Projection")
        # Create a test 3D point
        test_point_3d = (0.0, 1.0, 0.0)  # Directly ahead
        screen_point = ap.project_point(test_point_3d)
        print(f"  3D point {test_point_3d} projects to screen coordinates: {screen_point}")
        
        # Check if the point is projected to the center-middle of the image
        expected_x = test_image.shape[1] // 2
        expected_y = test_image.shape[0] // 2
        
        # Check if projection is close to expected
        x_diff = abs(screen_point[0] - expected_x)
        y_diff = abs(screen_point[1] - expected_y)
        
        if x_diff <= 5 and y_diff <= 5:
            print("  ✅ Projection looks correct - center point projects to middle of screen")
        else:
            print(f"  ❌ Projection may be off - expected around ({expected_x},{expected_y})")
            
        # Phase 4: Check 2D to 3D back-projection
        print("\nPhase 4: Testing 2D to 3D Conversion")
        screen_center = (test_image.shape[1] // 2, test_image.shape[0] // 2)
        back_projected = ap.reverse_project_point(screen_center)
        print(f"  Screen center {screen_center} back-projects to 3D point: {back_projected}")
        
        # Check if the back-projection has y value close to 1.0 (depth)
        if abs(back_projected[1] - 1.0) < 0.1:
            print("  ✅ Back-projection depth looks correct")
        else:
            print(f"  ❌ Back-projection depth may be off - expected y ≈ 1.0, got {back_projected[1]:.2f}")
        
        # Phase 5: Check image encoding for Gemini
        print("\nPhase 5: Testing Image Encoding for Gemini")
        # Try encoding a small section of the image to avoid huge output
        small_img = cv2.resize(test_image, (320, 240))
        
        # Calculate encoded size
        _, buffer = cv2.imencode('.jpg', small_img)
        encoded_size = len(buffer)
        
        print(f"  Test encoding of 320x240 image: {encoded_size} bytes")
        
        # Check full image size
        _, buffer = cv2.imencode('.jpg', test_image)
        encoded_size = len(buffer) / (1024 * 1024)  # Convert to MB
        
        print(f"  Full image encoded size: {encoded_size:.2f} MB")
        if encoded_size > 5:
            print("  ⚠️ Warning: Full image size is large for API transmission")
            print("     Consider resize or quality reduction before sending to Gemini")
        else:
            print("  ✅ Image size is reasonable for API transmission")
        
        # Create visualization with debug info
        print("\nCreating visualization with debug information...")
        debug_img = ap.visualize_coordinate_system(test_image.copy())
        
        # Add resolution info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, 
                   f"Image: {test_image.shape[1]}x{test_image.shape[0]}", 
                   (10, test_image.shape[0]-60), 
                   font, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_img, 
                   f"ActionProjector: {ap.image_width}x{ap.image_height}", 
                   (10, test_image.shape[0]-30), 
                   font, 0.7, (0, 255, 255), 2)
        
        # Save the debug image
        cv2.imwrite("resolution_check_debug.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        print(f"Debug visualization saved to resolution_check_debug.jpg")
        
        # Final conclusion
        print("\n=== CONCLUSION ===")
        print(f"Total check time: {time.time() - start_time:.2f} seconds")
        
        # Check if ActionProjector matches our test image
        if abs(ap.image_width - test_image.shape[1]) <= 5 and abs(ap.image_height - test_image.shape[0]) <= 5:
            print("✅ PASS: ActionProjector dimensions match the captured image")
        else:
            print(f"❌ FAIL: ActionProjector dimensions ({ap.image_width}x{ap.image_height}) " 
                 f"don't match captured image ({test_image.shape[1]}x{test_image.shape[0]})")
            
        print("\nRecommendation:")
        with mss.mss() as sct:
            m1 = sct.monitors[1]
            if abs(ap.image_width - m1['width']) > 5 or abs(ap.image_height - m1['height']) > 5:
                print(f"- Update ActionProjector dimensions to match your monitor: {m1['width']}x{m1['height']}")
            else:
                print("- Image processing pipeline appears to be correctly configured")
                
        print("\nNext steps:")
        print("1. Review the generated debug images")
        print("2. If any mismatches were identified, update ActionProjector.__init__")
        print("3. Run this check again to verify the fix")
            
    except Exception as e:
        print(f"Error during pipeline check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== RESOLUTION VERIFICATION TOOL ===")
    print("Checking monitor resolutions and image processing pipeline...")
    
    check_monitor_resolutions()
    check_processing_pipeline()
    
    print("\nCheck complete. Please review the results above and the generated debug images.")
    print("Press Ctrl+C to exit.")
    
    # Keep the script running so user can view the output
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...") 