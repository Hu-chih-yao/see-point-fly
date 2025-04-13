# Tello Drone Control System

A comprehensive system for controlling Tello drones with advanced computer vision capabilities, integrating Vision-Language Models (VLM) for automated control.

## Project Structure

```
VLM_Tello_integration/
├── camera_utils/         # Camera calibration and frame processing
│   └── camera_calibration.py  # Camera calibration tool
├── control_system/       # Drone control and flight management
│   ├── drone_controller.py    # Basic drone connectivity
│   └── keyboard_controller.py # Manual keyboard controls
├── safety/               # Safety monitoring and failsafe systems
│   └── emergency_handler.py   # Emergency monitoring and actions
├── detection/            # Object detection and tracking
├── config/               # Configuration files including calibration data
├── Documentations/       # Project documentation
│   ├── TelloDroneControlSystem_Architecture.md
│   ├── TelloDroneControlSystem_IncrementalSteps.md
│   ├── Phase1.1_DroneConnectivity.md
│   ├── Phase1.2_EmergencyControls.md
│   └── Phase2_VideoProcessingAndCalibration.md
├── Labs/                 # Laboratory files and testing code
├── main.py               # Main application entry point
└── README.md             # This file
```

## Current Implementation

### Phase 1.1: Basic Connectivity
- Establishing connection to the Tello drone
- Retrieving basic telemetry (battery, temperature, SDK version)
- Maintaining connection with keepalive signals
- Safe disconnection and error handling

### Phase 1.2: Emergency Controls
- Manual keyboard control for all flight axes (roll, pitch, yaw, throttle)
- Emergency stop functionality (via 'Q' key)
- Automated safety monitoring for critical conditions
- Variable speed modes for precise control

### Phase 2: Video Processing and Calibration (NEW)
- Robust video stream acquisition with error handling
- Frame preprocessing with color space conversion
- Stream stabilization with automatic warm-up period
- Interactive camera calibration tool with visual feedback
- Threshold adjustment for optimal pattern detection
- Calibration quality assessment and reporting
- Lens distortion correction with real-time preview
- Saving and loading of calibration parameters

## Keyboard Controls

```
=== Keyboard Controls ===
Takeoff:           't'
Land:              'l'
Emergency Stop:    'q'
Stop Movement:     'e'
Reset Velocities:  'r'
Forward/Backward:  'w'/'s'
Left/Right:        'a'/'d'
Up/Down:           'up'/'down' arrow keys
Yaw Left/Right:    'left'/'right' arrow keys
Get Height:        'h'
Get Battery:       'b'
Exit Program:      'ctrl+c'
```

The keyboard controller uses direct key mapping with sophisticated error handling and command rate limiting for smooth control response. All movement controls operate at configurable speeds, and safety mechanisms prevent command flooding.

## Requirements

- Python 3.6 or higher
- djitellopy
- opencv-python
- numpy
- pynput (for keyboard control)
- logging

## Installation

1. Ensure you have Python 3.6+ installed
2. Install required packages:

```bash
pip install djitellopy opencv-python numpy pynput
```

## Usage

### Main Application

1. Turn on your Tello drone
2. Connect your computer to the Tello's Wi-Fi network
3. Run the main script:

```bash
python main.py
```

This will:
- Connect to the drone
- Display basic information
- Start the emergency monitoring system
- Enable keyboard control
- Allow safe operation with emergency override capability

### Camera Calibration

1. Print a 6x9 chessboard pattern (7x10 squares)
2. Turn on your Tello drone
3. Connect your computer to the Tello's Wi-Fi network
4. Run the calibration tool:

```bash
python camera_utils/camera_calibration.py
```

5. Follow the on-screen instructions:
   - Press 'd' to toggle the debug view for threshold adjustment
   - Adjust the threshold slider until the chessboard is clearly detected
   - Press 'c' to capture when a chessboard is detected
   - Capture at least 15 samples from different angles
   - Press 'q' to finish and calculate calibration

The calibration files will be saved to `config/tello_camera_calibration.xml` and can be used by computer vision applications.

### Using Calibration Data in Applications

```python
import cv2

# Load calibration parameters
calibration_file = 'config/tello_camera_calibration.xml'
fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('dist_coeffs').mat()
fs.release()

# Undistort a frame
def undistort_frame(frame):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
```

## Safety Features

- Emergency stop button ('Q')
- Low battery auto-landing
- Critical battery emergency stop
- Maximum height limitation
- Fallback emergency mechanisms
- Response time tracking for diagnostics

## Troubleshooting

### Common Issues

1. **Connection Problems**
   - Make sure your computer is connected to the Tello's Wi-Fi network
   - Ensure the drone is powered on and has sufficient battery
   - Try restarting the drone if connection issues persist

2. **Keyboard Control Issues**
   - Verify terminal/console has focus when issuing keyboard commands
   - Check that no conflicting applications are capturing keyboard input
   - For some environments, running as administrator might be needed for keyboard capture

3. **Emergency Stop Behavior**
   - Be aware that emergency stop will immediately cut all motors
   - Ensure sufficient clearance below the drone when testing
   - Have a safe landing area available at all times

4. **Camera Calibration Issues**
   - If the chessboard isn't detected, try adjusting the threshold slider
   - Use the debug view ('d' key) to see the binary threshold image
   - Ensure good lighting and a clear view of the chessboard
   - Verify your chessboard matches the expected dimensions (6x9 inner corners)

5. **Video Stream Problems**
   - If video frames are corrupted, try reconnecting to the drone
   - Maintain a strong Wi-Fi connection for stable video streaming
   - Reduce sources of Wi-Fi interference for better performance
   - Wait for the warm-up period to complete for best video quality

## Development Plan

See the full development roadmap in the `Documentations` folder:
- [Architecture Overview](Documentations/TelloDroneControlSystem_Architecture.md)
- [Incremental Development Steps](Documentations/TelloDroneControlSystem_IncrementalSteps.md)
- [Phase 1.1: Drone Connectivity](Documentations/Phase1.1_DroneConnectivity.md)
- [Phase 1.2: Emergency Controls](Documentations/Phase1.2_EmergencyControls.md)
- [Phase 2: Video Processing and Calibration](Documentations/Phase2_VideoProcessingAndCalibration.md)

## Safety Notes

- Always operate the drone in a safe and open environment
- Maintain visual contact with the drone at all times
- Follow local regulations regarding drone operations
- The application includes safe shutdown mechanisms in case of unexpected errors
- Practice emergency landings in a safe environment before actual flights

## License

This project is intended for educational and research purposes only.

## Contributing

This is a research project in development. Contributions will be accepted in later phases. 