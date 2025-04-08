# Phase 2: Video Processing and Calibration

## Overview

Phase 2 focuses on enabling the Vision-Language Model integration by implementing robust video stream capture and advanced camera calibration capabilities. This phase establishes the foundation for all computer vision tasks in the project by ensuring high-quality video input and accurate spatial measurements.

## Key Components

### 2.1 Video Stream Processing

The video stream processing component connects to the Tello drone's camera feed, handles frame acquisition, and prepares frames for further analysis.

#### Implementation Details

- **Frame Acquisition**: Implemented in the drone controller with optimized initialization and error handling to ensure stable video stream connectivity
- **Color Space Conversion**: Automatic RGB to BGR conversion for OpenCV compatibility
- **Stream Stabilization**: Frame warm-up process that discards initial frames to stabilize the H.264 decoder
- **Robust Error Handling**: Comprehensive error detection and recovery for frame acquisition problems
- **Performance Metrics**: Real-time FPS measurement and visualization
- **Frame Validation**: Automatic detection and rejection of invalid or corrupted frames

#### Usage

The video stream is automatically initialized when connecting to the drone:

```python
from control_system.drone_controller import DroneController

# Initialize drone controller
drone = DroneController()
drone.connect()

# Video stream is automatically started during connection
frame = drone.get_frame()  # Get the latest frame
```

### 2.2 Camera Calibration

The camera calibration system enables accurate measurements by removing lens distortion effects from the captured frames.

#### Implementation Details

- **Interactive Calibration Tool**: User-friendly interface for camera calibration using chessboard pattern
- **Adaptive Detection**: Robust detection of calibration patterns using multiple threshold methods
- **Visual Feedback**: Real-time visualization of detected patterns and binary thresholding
- **Calibration Quality Assessment**: Calculation and interpretation of reprojection error
- **Parameter Storage**: Automatic saving and loading of calibration parameters in both XML and human-readable formats
- **Undistortion Preview**: Side-by-side comparison of original and undistorted frames
- **Configurable Pattern Size**: Support for different chessboard sizes (currently configured for 6x9 pattern)

#### Usage

The calibration process is divided into two parts:

1. **Calibration Data Collection**:

```bash
python camera_utils/camera_calibration.py
```

2. **Using Calibration in Applications**:

```python
import cv2
import numpy as np

# Load calibration parameters
calibration_file = 'config/tello_camera_calibration.xml'
fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeffs').mat()
fs.release()

# Undistort a frame
def undistort_frame(frame):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, newcameramtx
    )
    # Apply ROI if available
    x, y, w, h = roi
    if w > 0 and h > 0:
        undistorted = undistorted[y:y+h, x:x+w]
    return undistorted
```

## Design Considerations

### Video Stream Stabilization

Tello's video stream relies on H.264 encoding, which can produce corrupted frames during initialization or under poor Wi-Fi conditions. To mitigate this:

1. **Warm-up Period**: The system discards the initial frames to allow the decoder to stabilize
2. **Connection Retries**: Multiple connection attempts with exponential backoff
3. **Frame Validation**: Size and content validation before processing
4. **Color Space Normalization**: Consistent conversion to OpenCV's expected format

### Camera Calibration Approach

The calibration system balances ease of use with accuracy:

1. **Pattern Selection**: Chessboard pattern chosen for robust corner detection
2. **Multi-threshold Detection**: Uses both original and thresholded images to improve corner detection
3. **Sample Diversity**: Requires 15 samples from different viewpoints for comprehensive calibration
4. **Quality Metrics**: Calculates and displays reprojection error with quality interpretation
5. **Interactive Adjustment**: Threshold slider for fine-tuning detection parameters
6. **Debug Visualization**: Split-screen view for understanding binary thresholding effects

## Integration with Main System

The video processing and calibration components are designed to work seamlessly with the rest of the control system:

1. **Frame Acquisition**: The DroneController class has been extended to handle frame acquisition
2. **Optional Calibration**: Applications can choose to use raw or undistorted frames
3. **Error Propagation**: Frame acquisition errors are properly propagated to calling code
4. **Resource Management**: Proper cleanup of video resources on application exit

## Testing and Validation

### Video Stream Testing

- **Latency Testing**: Analysis of frame acquisition times
- **Stability Testing**: Long-running tests to verify stream stability
- **Error Recovery**: Validation of recovery after connection interruptions
- **Frame Rate Measurement**: Verification of consistent FPS under various conditions

### Calibration Testing

- **Before/After Comparison**: Visual verification of undistortion effects
- **Reprojection Error**: Quantitative measurement of calibration accuracy
- **Pattern Detection**: Testing under various lighting and distance conditions
- **Parameter Consistency**: Verification of calibration parameter stability across multiple calibrations

## Next Steps

With the video processing and calibration foundation in place, the project is now ready to proceed to:

1. **Object Detection**: Implementing YOLO-based detection for scene understanding
2. **Depth Estimation**: Adding depth perception capabilities
3. **Command Generation**: Integrating Vision-Language Models for automated command processing

## Conclusion

Phase 2 establishes the critical visual pipeline necessary for all subsequent computer vision tasks. The robust frame acquisition system and accurate camera calibration provide the foundation upon which object detection, depth estimation, and ultimately VLM-based drone control will be built. 