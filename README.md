# VLM Tello Integration System

An advanced drone navigation system that uses Vision Language Models (VLM) to control DJI Tello drones with natural language commands. The system supports both physical drone control and simulator environments with intelligent navigation capabilities.

## Overview

This system provides two main operation modes:

### 🚁 **Tello Mode** (Physical Drone Control)
Control real DJI Tello drones using natural language commands with two operational modes:
- **Adaptive Mode**: Precision navigation with advanced depth estimation
- **Obstacle Mode**: Enhanced safety with obstacle detection and avoidance

### 🎮 **Simulator Mode** (Virtual Environment) 
Test and develop navigation algorithms in a simulated environment using screen capture and keyboard controls.

## System Architecture

```
VLM Tello Integration System
├── Tello Mode (Physical Drone)
│   ├── 🎯 Adaptive Mode - Precision Navigation
│   └── 🛡️ Obstacle Mode - Safe Navigation  
└── Simulator Mode (Virtual Environment)
```


⚠️ **PROPRIETARY SOFTWARE** ⚠️

This software is proprietary and closed-source. All rights reserved.
No part of this software may be used, copied, modified, or distributed without express written permission.
See LICENSE file for details.

## Operational Modes

### 🎯 **Adaptive Mode** (Precision Navigation)
- **Best for**: Indoor navigation, precise positioning tasks
- **Features**: Advanced depth estimation, adaptive movement speed, precision control
- **AI Model**: Gemini 2.0 Flash (optimized for speed and accuracy)
- **Recording**: 3fps frame recording
- **Safety**: Standard error handling

### 🛡️ **Obstacle Mode** (Safe Navigation) 
- **Best for**: Complex environments, outdoor navigation, obstacle-rich areas
- **Features**: Obstacle detection, bounding box visualization, intensive keepalive system
- **AI Model**: Gemini 2.5 Pro Preview (advanced obstacle recognition)
- **Recording**: 10fps high-detail recording  
- **Safety**: Enhanced timeout protection, automatic safety landing

### 📋 **Technical Details**
For comprehensive technical documentation including system architecture, data flow analysis, and implementation details, refer to: [`documents/0904_unified_tello_system_architecture.md`](documents/0904_unified_tello_system_architecture.md)

## Key Features

- 🤖 **Natural Language Control**: Command drones using plain English
- 🧠 **AI-Powered Decision Making**: Google Gemini Vision for intelligent navigation
- 🎯 **Dual Navigation Modes**: Choose between precision and safety-focused operation
- 📹 **Real-Time Processing**: Live camera feed analysis and response
- ⚡ **Multi-Threaded Execution**: Responsive control with background processing
- 🎮 **Manual Override**: Instant keyboard control for safety
- 📊 **Comprehensive Logging**: Detailed operation tracking and debugging
- 🔄 **Automatic Recovery**: Intelligent error handling and safety systems

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

### Operational Mode Configuration

Configure your navigation mode in `config_tello.yaml`:

```yaml
# Choose your operational mode
operational_mode: "adaptive_mode"  # or "obstacle_mode"

# Processing configuration
command_loop_delay: 2  # seconds between processing cycles
```

### Mode Selection Guide

| Mode | Best For | AI Model | Safety Features |
|------|----------|----------|-----------------|
| `adaptive_mode` | Indoor precision tasks | Gemini 2.0 Flash | Standard error handling |
| `obstacle_mode` | Complex environments | Gemini 2.5 Pro | Enhanced safety + obstacle detection |

### Environment Setup

Create a `.env` file with your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

## Navigation Intelligence

### Adaptive Mode Intelligence
- **Dynamic Depth Estimation**: AI analyzes scene depth on a 1-10 scale for precise distance calculation
- **Non-Linear Movement Speed**: Slow and careful for close objects, fast and efficient for distant ones
- **Precision Targeting**: Direct targeting with intelligent depth-based movement scaling

### Obstacle Mode Intelligence  
- **Obstacle Detection**: AI identifies and maps obstacles with bounding box coordinates
- **Safe Path Planning**: Considers obstacles when selecting navigation points
- **Enhanced Safety**: Intensive keepalive system prevents disconnection during processing
- **Real-Time Adaptation**: Continuous obstacle monitoring and avoidance

## Usage

### Quick Start

1. **Connect** to your Tello's Wi-Fi network
2. **Configure** your preferred mode in `config_tello.yaml`
3. **Run** the system:

```bash
python main_tello.py
```

### Mode-Specific Usage

#### 🎯 **Adaptive Mode** (Precision Navigation)
Best for indoor, controlled environments:

```yaml
# config_tello.yaml
operational_mode: "adaptive_mode"
command_loop_delay: 2
```

```bash
# Standard precision flight
python main_tello.py

# With debug visualization
python main_tello.py --debug
```

**Example Commands:**
- "fly to the chair in front of you"
- "navigate to the center of the table"
- "move toward the window on the left"

#### 🛡️ **Obstacle Mode** (Safe Navigation)
Best for complex environments with obstacles:

```yaml
# config_tello.yaml  
operational_mode: "obstacle_mode"
command_loop_delay: 2
```

```bash
# Safe obstacle-aware flight
python main_tello.py

# With enhanced recording
python main_tello.py --record --record-session "outdoor_flight"
```

**Example Commands:**
- "navigate around the tree to reach the building"
- "fly through the doorway avoiding the walls"
- "move to the open area while avoiding obstacles"

### Advanced Options

```bash
# Test mode (static image)
python main_tello.py --test

# Debug mode (live camera feed + detailed logging)
python main_tello.py --debug

# Skip camera check (if camera issues)
python main_tello.py --skip-camera-check

# Record flight with custom session name
python main_tello.py --record --record-session "my_flight"
```

## System Modes Comparison

| Aspect | 🚁 Tello Mode | 🎮 Simulator Mode |
|--------|---------------|-------------------|
| **Environment** | Physical DJI Tello drone | Virtual screen-based simulation |
| **Input Source** | Live camera feed (720p) | Screen capture |
| **Control Method** | Direct RC commands | Keyboard simulation |
| **AI Models** | Dual modes (Gemini 2.0/2.5) | Single mode optimization |
| **Safety Systems** | Battery monitoring, keepalive | Standard error handling |
| **Setup Requirements** | Tello drone + Wi-Fi connection | Any computer with screen |
| **Best Use Cases** | Real flight testing, demonstrations | Algorithm development, testing |

### Tello Mode Advantages
- ✅ Real-world flight experience
- ✅ Advanced AI models (dual mode support)
- ✅ Enhanced safety systems
- ✅ Obstacle detection capabilities
- ✅ Physical depth perception

### Simulator Mode Advantages  
- ✅ No hardware requirements
- ✅ Safe development environment
- ✅ Rapid iteration and testing
- ✅ No battery or connection limitations

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

### Mode Selection Issues

**Problem**: Not sure which mode to use?
```yaml
# For indoor precision tasks
operational_mode: "adaptive_mode"

# For outdoor/complex environments  
operational_mode: "obstacle_mode"
```

### Connection Issues

- ✅ Ensure computer is connected to Tello's Wi-Fi network
- ✅ Tello battery should be >50% charged
- ✅ Keep Tello within 10 meters Wi-Fi range
- ✅ Restart Tello if connection fails

### Mode-Specific Issues

#### Adaptive Mode Troubleshooting
- **Slow Processing**: Normal for precision mode (2-5 seconds per command)
- **Depth Estimation**: Ensure good lighting and clear target objects
- **Movement Precision**: Works best in structured indoor environments

#### Obstacle Mode Troubleshooting  
- **Longer Processing**: Normal for enhanced mode (3-8 seconds per command)
- **Timeout Errors**: System includes 120-second timeout protection
- **Keepalive Messages**: Normal intensive keepalive logging during API calls
- **High Recording**: 10fps recording uses more storage space

### Performance Issues

- **API Timeouts**: Switch to `adaptive_mode` for faster processing
- **Memory Usage**: Obstacle mode uses ~100MB more RAM
- **Storage Space**: Obstacle mode records 3x more frames

### Safety Features

- **Automatic Landing**: System lands drone after 3-5 consecutive errors
- **Manual Override**: Use keyboard controls for immediate safety control  
- **Battery Monitoring**: Obstacle mode includes comprehensive battery warnings
