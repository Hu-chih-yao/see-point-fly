# Drone Spatial Navigation System

A system for controlling drone movements using high-level spatial commands and sophisticated 3D coordinate projection.

## Overview

This project enables natural 3D navigation for drones by projecting 2D screen coordinates into 3D space and generating appropriate control commands. It uses Gemini AI to interpret high-level spatial instructions and convert them into specific waypoints or actions.

## License Notice

⚠️ **PROPRIETARY SOFTWARE** ⚠️

This software is proprietary and closed-source. All rights reserved.
No part of this software may be used, copied, modified, or distributed without express written permission.
See LICENSE file for details.

## Features

- Screen capture for real-time drone view
- AI-powered decision making using Google Gemini
- Keyboard control interface
- Multi-threaded action execution
- Command queue system

## Prerequisites

- Python 3.8+
- Google Gemini API key
- [The Drone Racing League Simulator](https://store.steampowered.com/app/641780/The_Drone_Racing_League_Simulator/) (available on Steam)

## Simulator Setup

1. Install The Drone Racing League Simulator from Steam
2. Launch the simulator and go to Settings
3. Set difficulty to Easy
4. Enable "Height Automation" for better control
5. Keep the simulator window visible on your main monitor

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

Run the main script to start the navigation system:

```bash
python main.py
```

### Monitor Selection

The system now supports selecting which monitor to capture:

```bash
# Use primary monitor (recommended)
python main.py --monitor 1

# Use secondary monitor
python main.py --monitor 2

# List available monitors
python main.py --info
```

### Debug Mode

For diagnosing coordinate system and projection issues:

```bash
python main.py --debug
```

This will:
- Create a visualization of the coordinate system
- Check if monitor resolution matches expected dimensions
- Save the visualization to `coordinate_system_debug.jpg`

### Test Mode

Run with a static test image:

```bash
python main.py --test
```

## Screen Resolution

The system is configured for a screen resolution of **2880×1864**. If your monitor has a different resolution:

1. Run debug mode to check current dimensions:
```bash
python main.py --debug
```

2. Update in action_projector.py if needed:
```python
self.image_width = 2880   # Your monitor width
self.image_height = 1864  # Your monitor height
```

## Components

- **ActionProjector**: Handles 3D coordinate projection and spatial visualization
- **DroneController**: Manages drone movements and control commands
- **DroneActionSpace**: Defines the 3D action space and movement constraints

## Troubleshooting

### Monitor/Screen Issues

- If screen capture isn't working correctly, verify monitor selection with `python main.py --info`
- If projection points appear incorrect, make sure screen resolution matches ActionProjector dimensions

### Gemini API Issues

- Check your `.env` file contains a valid GEMINI_API_KEY
- Make sure you have internet connectivity