# Monitor Resolution Technical Findings

## Executive Summary

During development of the drone navigation system, we identified a critical issue affecting coordinate projection and image processing: **monitor resolution scaling**.

The core finding was that macOS Retina/HiDPI displays use a 2x scaling factor that creates a discrepancy between reported monitor dimensions and actual captured image dimensions. This issue, when unaddressed, causes several critical problems in the image processing pipeline.

This document details our investigation process, findings, and implemented solutions.

## Resolution Discrepancy

### Reported vs. Actual Dimensions

Through systematic testing, we identified the following:

| Monitor | Reported Resolution | Actual Captured Resolution | Scaling Factor |
|---------|--------------------|-----------------------------|----------------|
| Monitor 1 (Primary) | 1710×1107 | 3420×2214 | 2.0x |
| Monitor 2 | 2048×1152 | 2048×1152 | 1.0x |

**Key Finding**: Monitor 1 (the primary display) has a 2.0x resolution scaling factor, which is standard for macOS Retina/HiDPI displays.

### Investigation Process

We developed several diagnostic tools to systematically identify the issue:

1. Basic monitor information check:
   ```
   Total monitors: 3
   
   Monitor 0 (All monitors combined):
     Dimensions: 2102x2259
   
   Monitor 1:
     Dimensions: 1710x1107
     Screenshot dimensions: 3420x2214x4
     Scaling ratio: 2.00x (horizontal), 2.00x (vertical)
     ⚠️ Detected Retina/HiDPI display (2x scaling)
   
   Monitor 2:
     Dimensions: 2048x1152
     Screenshot dimensions: 2048x1152x4
     Scaling ratio: 1.00x (horizontal), 1.00x (vertical)
   ```

2. Detailed image dimension trace through processing pipeline:
   - Screen capture from Monitor 1 consistently produced 3420×2214 images
   - The `ActionProjector` class initially expected 1710×1107 images
   - This mismatch caused significant coordinate projection errors

## Impact on the System

The resolution mismatch caused several critical issues:

### 1. Coordinate Projection Errors

- 3D points were not projecting to expected 2D screen locations
- The center point (0, 1, 0) should project to the center of the image
- Instead, it projected to (855, 553) instead of (1710, 1107)

### 2. Visual Feedback Issues

- Visualization overlays were misaligned with the underlying image
- Points, vectors, and diagnostic information appeared in incorrect locations
- This made the visual debugging process extremely difficult

### 3. API Communication Problems

- Large 3420×2214 images were being sent to Gemini API
- These larger images required more bandwidth and processing time

## Solution Implementation

After thorough investigation, we implemented the "Raw Pixel Approach":

### 1. Updated ActionProjector Configuration

```python
class ActionProjector:
    def __init__(self):
        # Camera parameters updated to match actual screen capture resolution
        self.image_width = 3420    
        self.image_height = 2214   
        self.fov_horizontal = 108  # degrees
        self.fov_vertical = 108    # degrees
```

### 2. Improved Error Handling in Capture Function

```python
def capture_screen(monitor_index=1):
    try:
        with mss.mss() as sct:
            if monitor_index >= len(sct.monitors):
                monitor_index = 1
                
            monitor = sct.monitors[monitor_index]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    except Exception as e:
        print(f"Error capturing screen: {e}")
        blank = np.zeros((2214, 3420, 3), dtype=np.uint8)
        cv2.putText(blank, "Screen capture error", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return blank
```

### 3. Alternative Solution: Resolution Scaling

We also developed an alternative scaling approach that can be useful in certain scenarios:

```python
def capture_screen_fixed(monitor_index=1):
    """Returns image matching reported monitor dimensions"""
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
```

## Validation Results

After implementing the fixes, we ran comprehensive validation tests:

### 1. Raw Pixel Approach (Implemented)

```
✅ Captured image dimensions match ActionProjector configuration
✅ Encoded/decoded image dimensions match original
✅ Center projection is accurate: (1710, 1107) ≈ (1710, 1107)
✅ No issues detected in the image processing pipeline
```

### 2. Scaled Image Approach (Alternative)

```
Fixed image dimensions: 1710x1107
Raw image dimensions: 3420x2214
```

## Technical Explanation: Monitor Scaling

This issue is related to how macOS handles high-resolution displays:

1. **Physical Pixels vs. Logical Points**
   - Modern Retina displays have high pixel density (e.g., 220+ PPI)
   - macOS uses "points" as a device-independent measurement
   - On standard displays: 1 point = 1 pixel
   - On Retina displays: 1 point = 2×2 pixels

2. **API Behavior**
   - System APIs report resolution in logical points (e.g., 1710×1107)
   - Screen capture APIs capture actual pixels (e.g., 3420×2214)
   - This discrepancy is intentional but can cause issues in graphics applications

3. **Scaling Factors**
   - macOS supports scaling factors of 1.0x, 1.5x, and 2.0x
   - Our primary monitor uses 2.0x scaling
   - This is controlled in System Settings → Displays → Scaling options

## Recommendations

1. **For Development**
   - Always check monitor scaling when working on new systems
   - Use `tools/resolution/check_monitors.py` to verify monitor configuration
   - Adapt `ActionProjector` dimensions to match actual capture dimensions

2. **For Production**
   - Consider detecting scaling at runtime and adjusting dynamically
   - Add documentation for users about potential display scaling issues
   - Include diagnostic tools with the production package

3. **For Future Improvements**
   - Implement automatic scaling detection and adjustment
   - Consider adding a configuration option to choose between approaches
   - Optimize large image processing for better performance

## Appendices

### A. Test Results

| Test | Before Fix | After Fix |
|------|------------|-----------|
| Center point projection | (855, 553) | (1710, 1107) |
| Full pipeline check | Failed | Passed |
| API encoding | Successful but inefficient | Successful |

### B. Useful Commands

```bash
# Check monitor configuration
python tools/resolution/check_monitors.py

# Verify entire pipeline
python tools/resolution/check_pipeline.py

# Test image encoding for API
python tools/capture/check_encoding.py

# Generate resolution fixes
python tools/resolution/generate_fix.py
``` 