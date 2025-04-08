#!/usr/bin/env python3
"""
Image processing pipeline verification tool
Captures and analyzes images to identify resolution issues
"""

import cv2
import numpy as np
import base64
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from action_projector import ActionProjector
    from drone_controller import capture_screen
except ImportError:
    print("Error importing project modules. Make sure your PYTHONPATH includes the drone project root directory.")
    sys.exit(1)

# Directory for saving test images
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/screenshots'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_gemini_encoding():
    """Simulate image encoding process used for Gemini and analyze results"""
    print("\n=== GEMINI IMAGE ENCODING TEST ===")
    
    # 1. Capture an image 
    print("Capturing screen from monitor 1...")
    try:
        frame = capture_screen(monitor_index=1)
        print(f"Captured image dimensions: {frame.shape[1]}x{frame.shape[0]}")
        
        # Save the original captured image
        cv2.imwrite("test_captured.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Saved original captured image to test_captured.jpg")
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return
    
    # 2. Get ActionProjector's expected dimensions
    ap = ActionProjector()
    print(f"ActionProjector expects dimensions: {ap.image_width}x{ap.image_height}")
    
    # 3. Check if they match
    if frame.shape[1] == ap.image_width and frame.shape[0] == ap.image_height:
        print("✅ Captured image dimensions match ActionProjector configuration")
    else:
        print("❌ WARNING: Captured image dimensions don't match ActionProjector configuration")
        print(f"   Image: {frame.shape[1]}x{frame.shape[0]}, ActionProjector: {ap.image_width}x{ap.image_height}")
    
    # 4. Run image through the Gemini encoding process
    print("\nTesting Gemini encoding process...")
    # The exact encoding process used in action_projector.py
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    encoded_size = len(encoded_image) / (1024 * 1024)  # MB
    
    print(f"Encoded image size: {encoded_size:.2f} MB")
    
    # 5. Decode and verify the image dimensions
    decoded_buffer = base64.b64decode(encoded_image)
    decoded_image = cv2.imdecode(np.frombuffer(decoded_buffer, np.uint8), cv2.IMREAD_COLOR)
    
    print(f"Dimensions after encoding/decoding: {decoded_image.shape[1]}x{decoded_image.shape[0]}")
    
    # Save the encoded/decoded image to check visually
    cv2.imwrite("test_encoded_decoded.jpg", decoded_image)
    print("Saved encoded/decoded image to test_encoded_decoded.jpg")
    
    # 6. Check if dimensions are preserved
    if decoded_image.shape[1] == frame.shape[1] and decoded_image.shape[0] == frame.shape[0]:
        print("✅ Encoded/decoded image dimensions match original")
    else:
        print("❌ WARNING: Encoded/decoded image dimensions changed")
        print(f"   Original: {frame.shape[1]}x{frame.shape[0]}, Encoded: {decoded_image.shape[1]}x{decoded_image.shape[0]}")
    
    # 7. Simulate the point projection
    test_points = [
        (0.0, 1.0, 0.0),  # Center, forward 
        (1.0, 1.0, 0.0),  # Right, forward
        (-1.0, 1.0, 0.0), # Left, forward
        (0.0, 1.0, 1.0),  # Center, forward, up
        (0.0, 1.0, -1.0)  # Center, forward, down
    ]
    
    print("\nTesting 3D to 2D projection...")
    # Create a visualization of the projection
    viz_img = decoded_image.copy()
    
    for i, point_3d in enumerate(test_points):
        # Project using ActionProjector
        screen_point = ap.project_point(point_3d)
        
        # Draw on the image
        cv2.circle(viz_img, screen_point, 10, (0, 255, 0), -1)
        cv2.putText(viz_img, f"P{i+1}: {point_3d}", 
                   (screen_point[0]+15, screen_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"  3D point {point_3d} projects to screen point {screen_point}")
    
    # Save visualization
    cv2.imwrite("test_projection.jpg", viz_img)
    print("Saved projection visualization to test_projection.jpg")
    
    # 8. Check the expected center point
    center_point = ap.project_point((0.0, 1.0, 0.0))
    expected_center = (decoded_image.shape[1] // 2, decoded_image.shape[0] // 2)
    
    x_diff = abs(center_point[0] - expected_center[0])
    y_diff = abs(center_point[1] - expected_center[1])
    
    if x_diff <= 5 and y_diff <= 5:
        print(f"✅ Center projection is accurate: {center_point} ≈ {expected_center}")
    else:
        print(f"❌ Center projection is off: {center_point} vs expected {expected_center}")
    
    # 9. Final report
    print("\n=== FINAL REPORT ===")
    
    issues = []
    if frame.shape[1] != ap.image_width or frame.shape[0] != ap.image_height:
        issues.append("- Captured image dimensions don't match ActionProjector configuration")
    
    if decoded_image.shape[1] != frame.shape[1] or decoded_image.shape[0] != frame.shape[0]:
        issues.append("- Encoding/decoding changes image dimensions")
    
    if x_diff > 5 or y_diff > 5:
        issues.append("- Center point doesn't project to the middle of the image")
    
    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\nRecommendations:")
        print("1. Update ActionProjector.__init__ to use the actual monitor resolution")
        print("2. Review how the point projection is calculated")
    else:
        print("✅ No issues detected in the image processing pipeline")
        print("- Dimensions match throughout the pipeline")
        print("- Point projection appears to be working correctly")
    
    print("\nPlease review the generated test images for visual confirmation.")

if __name__ == "__main__":
    print("=== IMAGE PROCESSING PIPELINE VERIFICATION ===")
    print("Checking image dimensions throughout processing steps...")
    
    check_gemini_encoding()
    
    print("\nCheck complete.") 