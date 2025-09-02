# Tello Drone Spatial Navigation System

This is the Tello physical drone version of the Drone Spatial Navigation System, allowing you to control a DJI Tello drone using high-level spatial commands and AI-powered decision making.

## Overview

This implementation adapts the simulator-based drone navigation system to work with physical Tello drones. It maintains the same structure and command processing pipeline while adapting the input (camera) and output (control) interfaces to work with the Tello API.


⚠️ **PROPRIETARY SOFTWARE** ⚠️

This software is proprietary and closed-source. All rights reserved.
No part of this software may be used, copied, modified, or distributed without express written permission.
See LICENSE file for details.

## Features

- Direct Tello drone control via djitellopy
- Real-time camera feed processing from Tello
- AI-powered decision making using Google Gemini
- Advanced depth estimation for intelligent navigation
- Adaptive movement speed based on target distance
- Keyboard manual override system
- Multi-threaded action execution
- Command queue system
- Emergency stop functionality

## Prerequisites

- uv
- Python 3.13+
- Google Gemini API key
- DJI Tello drone
- Good Wi-Fi connection to the Tello

## Installation

1. Install dependencies and create Python virtual environment:
```bash
uv sync
```

2. Configure your environment:
- Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Configuration

The system can be configured using `config_tello.yaml`:

```yaml
mode: "single"  # or "waypoint"
command_loop_delay: 0  # seconds between actions
```

- **Mode**:
  - `single`: Generates one action at a time with intelligent depth estimation
  - `waypoint`: Generates a sequence of waypoints for more complex navigation

## Navigation Intelligence

The system uses several advanced techniques to navigate effectively:

1. **Dynamic Depth Estimation**:
   - Gemini Vision analyzes the scene to estimate target distance on a 1-10 scale
   - Close objects (occupying large portions of the frame) receive lower depth values
   - Distant objects (appearing small in frame) receive higher depth values

2. **Adaptive Movement Speed**:
   - Non-linear depth scaling adjusts movement speed based on target distance
   - Precision movements for close objects (slow, careful approach)
   - Efficient movements for distant objects (faster approach)
   - Prevents overshooting or collisions with nearby objects

## Usage

### Basic Usage

Make sure your computer is connected to the Tello's Wi-Fi network, then run:

```bash
python main_tello.py
```

### Debug Mode

For debugging with visual feedback:

```bash
python main_tello.py --debug
```

This will:
- Show the live Tello camera feed
- Print detailed logging information
- Display processing steps

### Test Mode

Run with a static test image:

```bash
python main_tello.py --test
```

## Differences from Simulator Version

1. **Input Source**:
   - Simulator: Screen capture via mss
   - Tello: Direct camera feed via djitellopy

2. **Control Mechanism**:
   - Simulator: Keyboard simulation via pynput
   - Tello: Direct RC commands via djitellopy API

3. **Startup Sequence**:
   - Simulator: No physical startup needed
   - Tello: Connect → Stream On → Takeoff sequence

4. **Safety Features**:
   - Additional safety checks for battery and connection
   - Automatic landing on error or connection loss

5. **Performance Considerations**:
   - Tello has lower resolution camera (720p)
   - Commands may need duration/speed calibration
   - Network latency affects control responsiveness

## Manual Override Controls

You can take manual control at any time by pressing these keys:

- `↑/↓` (Arrow keys): Forward/Backward
- `A/D`: Turn left/right
- `←/→` (Arrow keys): Roll left/right
- `W/S`: Up/Down
- `T`: Takeoff
- `L`: Land
- `E`: Emergency stop (stops all movement)

AI control will automatically resume when you release all keys.

## Troubleshooting

### Connection Issues

- Ensure your computer is connected to the Tello's Wi-Fi network
- Tello battery should be adequately charged (>50% recommended)
- Keep Tello within good Wi-Fi range (~10 meters)

### Camera Feed Issues

- If camera feed fails, the system will return a blank image
- Try restarting the Tello and reconnecting

### Control Issues

- If the Tello doesn't respond to commands, check battery level
- Ensure there's no interference from other nearby Wi-Fi networks
- Calibrate the Tello using the official Tello app if movement seems inconsistent
