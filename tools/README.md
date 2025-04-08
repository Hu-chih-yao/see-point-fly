# Drone Project Tools

This directory contains various diagnostic and utility tools for the drone project, organized into specific categories.

## Directory Structure

```
tools/
  ├── resolution/     # Tools for monitor resolution and scaling
  ├── capture/        # Tools for screen capture and image processing
  ├── debug/          # General debugging tools
  └── README.md       # This file
```

## Complete Setup Process

Setting up the drone project correctly involves ensuring that screen captures and coordinate projections work properly with your specific monitor configuration. Follow this step-by-step process to configure the project parameters:

### 1. Check Your Monitor Configuration

First, identify your monitor configuration and dimensions:

```bash
# Navigate to the project root directory
cd ~/path/to/drone

# Run the monitor check tool
python -m tools.resolution.check_monitors
```

Take note of your monitor dimensions, particularly for Monitor 1 (your primary display). You'll need these values when configuring the ActionProjector.

### 2. Test Screen Capture and Resolution Scaling

Many displays (especially Mac Retina/HiDPI displays) have scaling factors that cause screen captures to have different dimensions than what the system reports. Check if this applies to your setup:

```bash
# Run the fixed capture test
python -m tools.capture.fixed_capture
```

This will output the actual captured dimensions versus the reported monitor dimensions and save samples to the `output/screenshots/` directory. Look for the reported scaling ratio (typically 1x, 1.5x, or 2x).

### 3. Run the Resolution Fix Generator

If you detected a scaling mismatch in step 2, generate the necessary fixes:

```bash
# Generate resolution fixes
python -m tools.resolution.generate_fix
```

This tool automatically creates optimized capture functions that handle the scaling ratio correctly. The output will show the detected scaling factor and save fixes to the capture directory.

### 4. Verify the Entire Processing Pipeline

Run a comprehensive check of the image processing pipeline to ensure all components are working together:

```bash
# Check the full pipeline 
python -m tools.resolution.check_pipeline
```

This will verify that:
- Screen captures match the reported monitor dimensions
- ActionProjector is configured with the correct dimensions
- 3D point projection works correctly
- Image encoding for Gemini is functioning

### 5. Update ActionProjector Configuration (If Needed)

If steps 1-4 identified any dimension mismatches, you may need to update the ActionProjector parameters in `action_projector.py`:

1. Open `action_projector.py` in your editor
2. Locate the `__init__` method (around line 20)
3. Update the following parameters to match your primary monitor (from step 1):
   ```python
   def __init__(self, 
                image_width=YOUR_MONITOR_WIDTH,
                image_height=YOUR_MONITOR_HEIGHT,
                camera_matrix=None,
                dist_coeffs=None):
   ```

### 6. Test the Main Application

Finally, run the main application with the `--debug` flag to verify your configuration:

```bash
python main.py --debug
```

This will display the coordinate system visualization with the current settings. If everything looks correct, you're ready to run the full application.

## Tool Descriptions

### Resolution Tools

Tools in the `resolution/` directory help diagnose and fix monitor resolution issues:

- **`check_monitors.py`**: Simple tool to display all available monitors and their reported resolutions
  ```
  python -m tools.resolution.check_monitors
  ```

- **`check_pipeline.py`**: Comprehensive analysis of the entire image processing pipeline, verifying monitor resolutions, image dimensions, and projection accuracy
  ```
  python -m tools.resolution.check_pipeline
  ```

- **`generate_fix.py`**: Analyzes monitor scaling and generates fixed capture functions to handle HiDPI/Retina displays
  ```
  python -m tools.resolution.generate_fix
  ```

### Capture Tools

Tools in the `capture/` directory help with screen capture and image processing:

- **`check_encoding.py`**: Tests the image encoding/decoding process used for the Gemini API and validates that dimensions are preserved throughout processing
  ```
  python -m tools.capture.check_encoding
  ```

- **`fixed_capture.py`**: Provides specialized screen capture functions that account for HiDPI/Retina displays
  ```
  python -m tools.capture.fixed_capture
  ```

## Common Issues and Solutions

### 1. HiDPI/Retina Display Scaling

**Problem**: Screen captures have dimensions larger than the reported monitor resolution.

**Solution**: The fixed capture module automatically handles this by detecting the scaling ratio and resizing the captured images to match the expected dimensions. After running `generate_fix.py`, the main application will automatically use the fixed capture function.

### 2. ActionProjector Dimension Mismatch

**Problem**: The ActionProjector is configured with dimensions that don't match your monitor.

**Solution**: Update the ActionProjector constructor parameters to match your primary monitor's dimensions, as reported by `check_monitors.py`.

### 3. Multiple Monitor Confusion

**Problem**: When using multiple monitors, the application captures the wrong screen.

**Solution**: Specify which monitor to capture using the `--monitor` flag:
```bash
python main.py --monitor 2  # Use monitor index 2
```

### 4. Point Projection Issues

**Problem**: 3D points don't project correctly to expected screen coordinates.

**Solution**: Run `check_pipeline.py` to verify the entire processing pipeline and look for specific errors in the coordinate projection tests. You may need to adjust the FOV parameters in ActionProjector.

## Usage in Custom Environments

If you're running in a custom environment (virtual machine, remote desktop, etc.), follow these additional steps:

1. Check monitor details: `python -m tools.resolution.check_monitors`
2. Run the resolution fix generator: `python -m tools.resolution.generate_fix`
3. Update the ActionProjector dimensions in `action_projector.py`
4. Verify with: `python main.py --info` (shows monitor information)
5. Test with: `python main.py --debug` (shows coordinate visualization)

For more detailed information about monitor resolution issues and fixes, see the documentation in the `documents/` directory. 