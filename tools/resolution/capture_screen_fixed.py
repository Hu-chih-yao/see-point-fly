#!/usr/bin/env python3
'''
Resolution-fixed screen capture function
Generated to handle 2.00x display scaling
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
                print(f"Warning: Monitor index {monitor_index} out of range. Using main monitor (1).")
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
        print(f"Error capturing screen: {e}")
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
        print(f"Error capturing screen: {e}")
        return None

if __name__ == "__main__":
    # Test the function
    print("Testing resolution-fixed screen capture...")
    
    # Capture with both functions
    start_time = time.time()
    img_fixed = capture_screen_fixed(1)
    img_raw = capture_screen_raw(1)
    
    print(f"Fixed image dimensions: {img_fixed.shape[1]}x{img_fixed.shape[0]}")
    if img_raw is not None:
        print(f"Raw image dimensions: {img_raw.shape[1]}x{img_raw.shape[0]}")
    
    # Save for comparison
    cv2.imwrite("capture_fixed.jpg", cv2.cvtColor(img_fixed, cv2.COLOR_RGB2BGR))
    if img_raw is not None:
        cv2.imwrite("capture_raw.jpg", cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR))
        
    print(f"Capture completed in {time.time() - start_time:.2f} seconds")
    print("Images saved to capture_fixed.jpg and capture_raw.jpg")
