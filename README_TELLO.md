# Tello Drone Spatial Navigation System

This is the Tello physical drone version of the Drone Spatial Navigation System, allowing you to control a DJI Tello drone using high-level spatial commands and AI-powered decision making.

## Overview

This implementation adapts the simulator-based drone navigation system to work with physical Tello drones. It maintains the same structure and command processing pipeline while adapting the input (camera) and output (control) interfaces to work with the Tello API.

## Features

- Direct Tello drone control via djitellopy
- Real-time camera feed processing from Tello
- AI-powered decision making using Google Gemini
- Keyboard manual override system
- Multi-threaded action execution
- Command queue system
- Emergency stop functionality

## Prerequisites

- Python 3.8+
- Google Gemini API key
- DJI Tello drone
- Good Wi-Fi connection to the Tello

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment:
- Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Configuration

The system can be configured using `config.yaml`:

```yaml
mode: "single"  # or "waypoint"
command_loop_delay: 0  # seconds between actions
```

- **Mode**:
  - `single`: Generates one action at a time
  - `waypoint`: Generates a sequence of waypoints for more complex navigation

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