# Screen Capture System Documentation

This document provides detailed information about the screen capture system used in the drone project, including resolution issues discovered and their solutions.

## Table of Contents

1. [Overview](#overview)
2. [Resolution Issues](#resolution-issues)
3. [Capture Workflow](#capture-workflow)
4. [Tools](#tools)
5. [Testing & Validation](#testing--validation)
6. [Troubleshooting](#troubleshooting)

## Overview

The screen capture system is responsible for capturing the drone simulator screen and processing images for computer vision analysis. It uses the Python `mss` library for fast screen capture and processes images through multiple stages before sending them to Gemini for analysis.

### Key Components

1. **Screen Capture Module** - Captures frames from the simulator screen
2. **Image Preprocessing** - Converts colorspace and prepares images for analysis
3. **Coordinate Projection** - Maps between 2D screen coordinates and 3D world space
4. **API Integration** - Handles sending images to Gemini for processing

## Resolution Issues

During development, we discovered significant issues with monitor resolution and scaling:

### HiDPI / Retina Display Scaling

The primary issue identified is **2.0x scaling** on macOS Retina displays:
- System reports monitor resolution as **1710×1107**
- Actual captured images have resolution **3420×2214**
- This 2x scaling is a standard feature of macOS HiDPI displays

### Impact on Coordinate Projection

This scaling mismatch caused several problems:
- The `ActionProjector` class expected images to match the reported monitor dimensions
- This led to incorrect coordinate calculations when projecting between 2D and 3D space
- Points weren't projecting to expected screen locations

### Solution Approach

Two potential solutions were identified:

1. **Raw Pixel Approach (Implemented)**
   - Configure `ActionProjector` to use the actual pixel dimensions (3420×2214)
   - Keep the full resolution for maximum detail in image processing
   - Maintain raw captured images without scaling/resizing

2. **Scaled Image Approach (Alternative)**
   - Downscale captured images to match reported monitor dimensions
   - Configure `ActionProjector` to use reported dimensions (1710×1107)
   - Improve performance at the cost of some image detail

## Capture Workflow

The current screen capture workflow consists of these steps:

1. **Frame Acquisition**
   ```python
   def capture_screen(monitor_index=1):
       with mss.mss() as sct:
           monitor = sct.monitors[monitor_index]
           screenshot = sct.grab(monitor)
           img = np.array(screenshot)
           return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
   ```

2. **API Preparation**
   ```python
   # Convert image to base64 for API transmission
   _, buffer = cv2.imencode('.jpg', image)
   encoded_image = base64.b64encode(buffer).decode('utf-8')
   ```

3. **Coordinate Projection**
   ```python
   def project_point(self, point_3d):
       # Convert 3D point to 2D screen coordinates
       x, y, z = point_3d
       # ... projection calculations ...
       screen_x = int(center_x + x_projected)
       screen_y = int(center_y - z_projected)
       return (screen_x, screen_y)
   ```

4. **Visualization**
   - Draw projected points, coordinate systems, and debug information
   - Save images for analysis and debugging

## Tools

Several diagnostic and fix tools have been created to address these issues:

### Resolution Checking

- **Check Monitors Tool** (`tools/resolution/check_monitors.py`)
  - Simple tool to display all available monitors and their reported resolutions
  - Use with: `python tools/resolution/check_monitors.py`

- **Check Pipeline Tool** (`tools/resolution/check_pipeline.py`)
  - Comprehensive analysis of the entire image processing pipeline
  - Verifies monitor resolutions, image dimensions, and projection accuracy
  - Use with: `python tools/resolution/check_pipeline.py`

### Screen Capture Tools

- **Check Encoding Tool** (`tools/capture/check_encoding.py`)
  - Tests the image encoding/decoding process used with Gemini API
  - Validates that dimensions are preserved throughout processing
  - Use with: `python tools/capture/check_encoding.py`

- **Fixed Capture Tool** (`tools/capture/fixed_capture.py`)
  - Provides two capture functions:
    - `capture_screen_fixed()` - Returns images scaled to match reported monitor resolution
    - `capture_screen_raw()` - Returns full-resolution images
  - Use with: `python tools/capture/fixed_capture.py`

### Resolution Fixing

- **Generate Fix Tool** (`tools/resolution/generate_fix.py`)
  - Analyzes monitor scaling and generates the fixed capture functions
  - Creates the necessary code to handle HiDPI/Retina displays
  - Use with: `python tools/resolution/generate_fix.py`

## Testing & Validation

A systematic approach was used to identify and fix resolution issues:

1. **Monitor Analysis**
   - All available monitors were checked for their reported and actual dimensions
   - Scaling ratios were calculated to detect HiDPI/Retina displays

2. **Pipeline Verification**
   - Images were traced through the entire processing pipeline
   - Dimension checks performed at each step
   - Coordinate projections tested with known points

3. **Visualization Tests**
   - Test images generated for visual confirmation
   - Coordinate systems rendered to verify accuracy
   - 3D/2D projections visualized for validation

## Troubleshooting

### Common Issues

1. **Image Dimension Mismatch**
   - **Symptom**: ActionProjector configured with different dimensions than captured images
   - **Check**: Run `python tools/resolution/check_pipeline.py`
   - **Fix**: Update ActionProjector.__init__ dimensions to match actual captured resolution

2. **Incorrect Projection**
   - **Symptom**: Points don't project to expected screen locations
   - **Check**: Run `python tools/capture/check_encoding.py`
   - **Fix**: Verify coordinate system and projection formulas

3. **API Transmission Errors**
   - **Symptom**: Errors when sending images to Gemini
   - **Check**: Check encoded image size with `tools/capture/check_encoding.py`
   - **Fix**: Consider resizing large images before encoding

### Testing on Different Systems

When deploying on a new system:

1. Check monitor dimensions and scaling:
   ```bash
   python tools/resolution/check_monitors.py
   ```

2. Update ActionProjector if necessary:
   ```python
   # In action_projector.py
   self.image_width = [width]    # Update to match actual pixel dimensions
   self.image_height = [height]  # Update to match actual pixel dimensions
   ```

3. Verify fixed configuration:
   ```bash
   python tools/resolution/check_pipeline.py
   ``` 