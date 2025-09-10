# Unified Tello System Architecture - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [Operational Modes](#operational-modes)
4. [Component Architecture](#component-architecture)
5. [Data Flow Analysis](#data-flow-analysis)
6. [Configuration System](#configuration-system)
7. [API Interfaces](#api-interfaces)
8. [Technical Specifications](#technical-specifications)
9. [Implementation Details](#implementation-details)
10. [Performance Characteristics](#performance-characteristics)

## System Overview

The SPF (See, Point, Fly) System consists of two independent drone navigation platforms:

1. **Tello Mode**: Controls physical DJI Tello drones with two operational modes (Adaptive and Obstacle)
2. **Simulator Mode**: Provides virtual drone simulation for development and testing

Each mode has its own entry point, controller, and supporting components, allowing for specialized functionality while maintaining architectural consistency.

### High-Level Architecture

```
SPF (See, Point, Fly) System Architecture
├── Entry Point: `spf` command
│   └── Main Router: src/spf/main.py
├── Tello Mode (Physical Drone)
│   ├── Entry: `spf tello`
│   ├── Main Module: src/spf/tello_main.py
│   ├── Operational Modes:
│   │   ├── adaptive_mode (Precision Navigation)
│   │   └── obstacle_mode (Safe Navigation)
│   └── Core Components:
│       ├── Controller: src/spf/controllers/tello_controller.py
│       ├── Projector: src/spf/projectors/action_projector.py
│       ├── Space: src/spf/spaces/drone_space.py
│       └── Client: src/spf/clients/vlm_client.py
└── Simulator Mode (Virtual Environment)
    ├── Entry: `spf sim`
    ├── Main Module: src/spf/simulator_main.py
    ├── Single Operation Mode
    └── Core Components:
        ├── Controller: src/spf/controllers/sim_controller.py
        ├── Projector: src/spf/projectors/action_projector_sim.py
        ├── Space: src/spf/spaces/drone_space_sim.py
        └── Client: src/spf/clients/vlm_client.py
```

## Architecture Design

### File Organization

#### Tello Mode Files
- **Entry Point**: `spf tello` command (via `src/spf/main.py`)
- **Main Module**: `src/spf/tello_main.py`
- **Core Components**:
  - `src/spf/controllers/tello_controller.py` - Physical drone control and communication
  - `src/spf/projectors/action_projector.py` - 2D/3D projection and Gemini integration
  - `src/spf/spaces/drone_space.py` - Action space and command conversion
- **Configuration**: `config_tello.yaml`

#### Simulator Mode Files
- **Entry Point**: `spf sim` command (via `src/spf/main.py`)
- **Main Module**: `src/spf/simulator_main.py`
- **Core Components**:
  - `src/spf/controllers/sim_controller.py` - Virtual drone control via keyboard simulation
  - `src/spf/projectors/action_projector_sim.py` - Screen-based projection and Gemini integration
  - `src/spf/spaces/drone_space_sim.py` - Simulator-specific action space
- **Configuration**: `config_sim.yaml`

### Core Design Principles

1. **Separate System Architecture**: Two independent systems with their own entry points and components
2. **Mode-Driven Configuration**: Tello mode supports dual operational modes via configuration
3. **Specialized Components**: Each system has optimized components for its environment
4. **Safety-First Design**: Multiple safety layers and error recovery mechanisms

### Component Hierarchy

#### Tello Mode Architecture
```
┌─────────────────────────────────────────────────┐
│                    SPF Command                  │
│                   (spf tello)                   │
└─────────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │              Main Entry           │
    │          (src/spf/main.py)        │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │            Tello Main             │
    │       (src/spf/tello_main.py)     │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────────────┐
    │           TelloController                 │
    │ (src/spf/controllers/tello_controller.py) │
    └─────────────────┬─────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
    ┌─────▼─────┐ ┌───▼────┐ ┌───▼─────┐
    │  Action   │ │ Drone  │ │ VLM     │
    │ Projector │ │ Space  │ │ Client  │
    │projectors/│ │spaces/ │ │clients/ │
    └───────────┘ └────────┘ └─────────┘
```

#### Simulator Mode Architecture
```
┌─────────────────────────────────────────────────┐
│                    SPF Command                  │
│                    (spf sim)                    │
└─────────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │              Main Entry           │
    │          (src/spf/main.py)        │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │          Simulator Main           │
    │     (src/spf/simulator_main.py)   │
    └─────────────────┬─────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │            SimController                │
    │ (src/spf/controllers/sim_controller.py) │
    └─────────────────┬───────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
    ┌─────▼─────┐ ┌───▼────┐ ┌───▼─────┐
    │ Action    │ │ Drone  │ │ VLM     │
    │Projector  │ │ Space  │ │ Client  │
    │ Sim       │ │ Sim    │ │clients/ │
    │projectors/│ │spaces/ │ │         │
    └───────────┘ └────────┘ └─────────┘
```

## Operational Modes

### Tello Mode Operational Modes

#### 1. Adaptive Mode (`adaptive_mode`)
- **Purpose**: Precision navigation with advanced depth estimation
- **Model**: Gemini 2.0 Flash
- **Focus**: Accurate positioning and depth-aware movement
- **Use Case**: Indoor navigation, precise positioning tasks

**Technical Characteristics**:
```yaml
Model: gemini-2.0-flash
Depth_Processing: Non-linear scaling (1-10 → 0.5-6.0)
Frame_Recording: 3fps
Keepalive_System: Disabled
Error_Tolerance: 5 consecutive errors
Timeout_Protection: None
Prompt_Strategy: Depth estimation with precision focus
```

**Depth Estimation Algorithm**:
```python
def calculate_adjusted_depth(gemini_depth):
    base = (gemini_depth / 10.0)**1.8 * 6.0
    adjusted_depth = max(0.5, base)
    return adjusted_depth
```

#### 2. Obstacle Mode (`obstacle_mode`)
- **Purpose**: Safe navigation with obstacle detection and avoidance
- **Model**: Gemini 2.5 Pro
- **Focus**: Safety and obstacle awareness
- **Use Case**: Complex environments, outdoor navigation

**Technical Characteristics**:
```yaml
Model: gemini-2.5-pro
Obstacle_Detection: Bounding box identification
Frame_Recording: 10fps
Keepalive_System: Intensive (1s intervals during API calls)
Error_Tolerance: 3 consecutive errors
Timeout_Protection: 120s with threaded processing
Prompt_Strategy: Obstacle-aware navigation
```

**Obstacle Detection Format**:
```json
{
    "point": [y, x],
    "label": "action description",
    "obstacles": [
        {
            "bounding_box": [ymin, xmin, ymax, xmax],
            "label": "obstacle_description"
        }
    ]
}
```

## Component Architecture

### 1. ActionProjector Class

**Initialization Signature**:
```python
def __init__(self,
             image_width=960,
             image_height=720,
             camera_matrix=None,
             dist_coeffs=None,
             mode="adaptive_mode")
```

**Mode-Specific Initialization**:
- **Model Selection**: Dynamic model selection based on operational mode
- **Prompt Engineering**: Different prompt strategies for each mode
- **Processing Pipeline**: Mode-specific JSON parsing and response handling

**Key Methods**:
```python
# Core processing methods
get_gemini_points(image, instruction, tello_controller=None)
_get_single_action(image, instruction, tello_controller=None)

# Projection methods
project_point(point_3d) -> Tuple[int, int]
reverse_project_point(point_2d, depth=2.0) -> Tuple[float, float, float]

# Utility methods
calculate_adjusted_depth(gemini_depth) -> float
visualize_coordinate_system(image=None) -> np.ndarray
```

### 2. TelloController Class

**Initialization Signature**:
```python
def __init__(self, mode="adaptive_mode")
```

**Mode-Specific Components**:
- **Keepalive System**: Obstacle mode only
- **Frame Recording**: Variable FPS based on mode
- **Error Handling**: Different tolerance levels per mode
- **Status Monitoring**: Enhanced monitoring in obstacle mode

**Key Methods**:
```python
# Core control methods
process_spatial_command(frame, instruction, mode="single")
_execute_spatial_action(action, quiet=False)

# Keepalive management (obstacle_mode only)
start_intensive_keepalive()
stop_intensive_keepalive()
check_drone_status()

# Safety methods
takeoff()
land()
stop()
```

### 3. DroneActionSpace Class

**Enhanced ActionPoint**:
```python
@dataclass
class ActionPoint:
    dx: float
    dy: float
    dz: float
    action_type: str
    screen_x: float = 0.0
    screen_y: float = 0.0
    detected_obstacles: list = None  # obstacle_mode only
```

## Data Flow Analysis

### Tello Mode Data Flows

#### Adaptive Mode Data Flow (Tello)
```
 ┌─────────────────┐    ┌───────────────┐    ┌──────────────────┐
 │ Tello Camera    │───▶│   Frame       │───▶│    Gemini        │
 │   Capture       │    │ Processing    │    │   2.0 Flash      │
 │(controllers/    │    │(tello_main.py)│    │ (projectors/     │
 │tello_controller)│    │               │    │action_projector) │
 └─────────────────┘    └───────────────┘    └──────────┬───────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────▼─────────┐
│   Drone         │◀───│   3D Point      │◀───│ Depth Estimation  │
│  Commands       │    │ Projection      │    │   JSON Parse      │
│(controllers/    │    │ (projectors/    │    │ (projectors/      │
│tello_controller)│    │action_projector)│    │action_projector)  │
└─────────────────┘    └─────────────────┘    └───────────────────┘
```

#### Obstacle Mode Data Flow (Tello)
```
  ┌─────────────────┐    ┌───────────────┐    ┌─────────────────┐
  │ Tello Camera    │───▶│   Frame       │───▶│  Keepalive      │
  │   Capture       │    │ Processing    │    │  Activation     │
  │(controllers/    │    │(tello_main.py)│    │(controllers/    │
  │tello_controller)│    │               │    │tello_controller)│
  └─────────────────┘    └───────────────┘    └─────────┬───────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────▼───────┐
│   Safety        │◀───│   Obstacle      │◀───│   Gemini        │
│  Navigation     │    │  Detection      │    │  2.5 Pro        │
│(controllers/    │    │ (projectors/    │    │ (projectors/    │
│tello_controller)│    │action_projector)│    │action_projector)│
└─────────────────┘    └─────────────────┘    └─────────┬───────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────▼───────┐
│  Keepalive      │    │   Drone         │    │  Timeout &      │
│Deactivation     │    │  Commands       │    │ Error Handling  │
│(controllers/    │    │(controllers/    │    │(tello_main.py)  │
│tello_controller)│    │tello_controller)│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Simulator Mode Data Flow

```
  ┌───────────────┐    ┌──────────────┐    ┌─────────────────┐
  │   Screen      │───▶│   Frame      │───▶│    Gemini       │
  │  Capture      │    │ Processing   │    │   2.5 Pro       │
  │(controllers/  │    │(simulator_   │    │ (projectors/    │
  │sim_controller)│    │  main.py)    │    │action_projector_│
  └───────────────┘    └──────────────┘    │      sim)       │
                                           └─────────┬───────┘
                                                     │
┌───────────────┐    ┌────────────────┐    ┌─────────▼───────┐
│  Keyboard     │◀───│   Action       │◀───│   Obstacle      │
│  Commands     │    │ Conversion     │    │  Detection      │
│(controllers/  │    │  (spaces/      │    │ (projectors/    │
│sim_controller)│    │drone_space_sim)│    │action_projector_│
└───────────────┘    └────────────────┘    │      sim)       │
                                           └─────────────────┘
```

## Configuration System

### Tello Mode Configuration

#### Primary Configuration File: `config_tello.yaml`

```yaml
# Operational Mode Configuration
operational_mode: "adaptive_mode"  # or "obstacle_mode"

# Processing Configuration
command_loop_delay: 2  # seconds between processing cycles

# Advanced Configuration (mode-specific)
# These are automatically handled based on operational_mode
```

#### Configuration Loading Pipeline

1. **Main Controller** (`spf tello` command) loads `config_tello.yaml`
2. **Mode Detection** from `operational_mode` parameter
3. **Component Initialization** with mode-specific parameters
4. **Runtime Behavior** adaptation based on mode

### Simulator Mode Configuration

#### Primary Configuration File: `config_sim.yaml`

```yaml
# Processing Configuration
command_loop_delay: 0  # seconds between processing cycles

# Simulator-specific settings
# Single operational mode (no mode switching)
```

#### Configuration Loading Pipeline

1. **Main Controller** (`spf sim` command) loads `config_sim.yaml`
2. **Standard Initialization** with simulator-specific parameters
3. **Runtime Behavior** optimized for virtual environment

### Environment Configuration: `.env`

```env
GEMINI_API_KEY=your_api_key_here
```

## API Interfaces

### Tello Mode APIs

#### ActionProjector API (`src/spf/projectors/action_projector.py`)

```python
class ActionProjector:
    def __init__(self, mode="adaptive_mode"):
        """Initialize with operational mode (adaptive_mode/obstacle_mode)"""

    def get_gemini_points(self,
                         image: np.ndarray,
                         instruction: str,
                         tello_controller=None) -> List[ActionPoint]:
        """
        Main processing method that handles both operational modes

        Args:
            image: Input camera frame from Tello
            instruction: Natural language command
            tello_controller: Controller reference for keepalive (obstacle_mode)

        Returns:
            List containing single ActionPoint with mode-specific processing
        """
```

#### TelloController API (`src/spf/controllers/tello_controller.py`)

```python
class TelloController:
    def __init__(self, mode="adaptive_mode"):
        """Initialize with operational mode"""

    def process_spatial_command(self,
                               current_frame: np.ndarray,
                               instruction: str,
                               mode: str = "single") -> str:
        """
        Process spatial command with mode-specific handling

        Args:
            current_frame: Camera frame from Tello
            instruction: Natural language instruction
            mode: Processing mode (always "single" in current implementation)

        Returns:
            String description of executed action
        """
```

### Simulator Mode APIs

#### ActionProjectorSim API (`src/spf/projectors/action_projector_sim.py`)

```python
class ActionProjector:
    def __init__(self, image_width=3420, image_height=2214):
        """Initialize for simulator screen resolution"""

    def get_gemini_points(self,
                         image: np.ndarray,
                         instruction: str) -> List[ActionPoint]:
        """
        Process screen capture for simulator navigation

        Args:
            image: Screen capture frame
            instruction: Natural language command

        Returns:
            List containing single ActionPoint with obstacle detection
        """
```

#### SimController API (`src/spf/controllers/sim_controller.py`)

```python
class SimController:
    def process_spatial_command(self,
                               current_frame: np.ndarray,
                               instruction: str) -> str:
        """
        Process spatial command for virtual drone

        Args:
            current_frame: Screen capture frame
            instruction: Natural language instruction

        Returns:
            String description of keyboard commands executed
        """
```

## Technical Specifications

### Performance Characteristics

#### Tello Mode Performance

| Metric | Adaptive Mode | Obstacle Mode |
|--------|---------------|---------------|
| **API Latency** | 2-5 seconds | 3-8 seconds |
| **Frame Rate** | 20fps (input) | 20fps (input) |
| **Recording Rate** | 3fps | 10fps |
| **Memory Usage** | ~200MB | ~250MB |
| **CPU Usage** | Medium | Medium-High |
| **Network Usage** | Low (Tello Wi-Fi) | Medium (Tello Wi-Fi) |

#### Simulator Mode Performance

| Metric | Simulator Mode |
|--------|----------------|
| **API Latency** | 3-8 seconds |
| **Screen Capture Rate** | 20fps (configurable) |
| **Recording Rate** | N/A (no frame recording) |
| **Memory Usage** | ~180MB |
| **CPU Usage** | Medium |
| **Network Usage** | Low (API only) |

### Hardware Requirements

#### Tello Mode Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Network**: Stable Wi-Fi connection to Tello drone
- **Storage**: 2GB free space for frame storage (10fps recording in obstacle_mode)
- **Hardware**: DJI Tello drone with charged battery

#### Simulator Mode Requirements
- **CPU**: Multi-core processor (2+ cores minimum)
- **RAM**: 2GB minimum, 4GB recommended
- **Network**: Internet connection for Gemini API
- **Storage**: 500MB free space for action visualizations
- **Display**: Screen/monitor for simulation environment

### Software Dependencies

#### Tello Mode Dependencies
```yaml
Core_Dependencies:
  - Python: ">=3.13"
  - djitellopy: "Latest"          # Tello drone communication
  - google-generativeai: "Latest" # Gemini API
  - opencv-python: "Latest"       # Computer vision
  - numpy: "Latest"               # Numerical operations
  - pynput: "Latest"              # Manual keyboard override
  - python-dotenv: "Latest"       # Environment variables

Visualization_Dependencies:
  - matplotlib: "For 3D plotting and visualization"
  - mpl_toolkits: "For 3D coordinate system visualization"
```

#### Simulator Mode Dependencies
```yaml
Core_Dependencies:
  - Python: ">=3.13"
  - mss: "Latest"                 # Screen capture
  - google-generativeai: "Latest" # Gemini API
  - opencv-python: "Latest"       # Computer vision
  - numpy: "Latest"               # Numerical operations
  - pynput: "Latest"              # Keyboard simulation
  - python-dotenv: "Latest"       # Environment variables

Visualization_Dependencies:
  - matplotlib: "For action visualization"
  - mpl_toolkits: "For coordinate system debugging"
```

## Implementation Details

### Keepalive System (Obstacle Mode)

**Purpose**: Prevent Tello automatic landing during long API calls

**Implementation**:
```python
def _keepalive_loop(self):
    while self.running and self.keepalive_active:
        if self.tello.is_flying and not self.manual_control_active:
            self.tello.send_keepalive()
            if self.intensive_keepalive:
                time.sleep(1)  # Intensive mode
            else:
                time.sleep(5)  # Normal mode
```

**Activation Strategy**:
- Normal keepalive: 5-second intervals
- Intensive keepalive: 1-second intervals during API calls
- Automatic activation/deactivation around Gemini API calls

### Error Recovery System

**Adaptive Mode**:
- 5 consecutive error tolerance
- Standard error logging
- Graceful degradation

**Obstacle Mode**:
- 3 consecutive error tolerance (stricter)
- Enhanced error logging with timestamps
- API timeout protection (120 seconds)
- Automatic safety landing on critical errors

### Depth Processing Algorithm

**Non-linear Depth Scaling** (Adaptive Mode):
```python
def calculate_adjusted_depth(self, gemini_depth):
    """
    Non-linear depth adjustment:
    - Close objects (1-3): Slow, careful movements
    - Far objects (7-10): Fast, efficient movements
    """
    base = (gemini_depth / 10.0)**1.8 * 6.0
    adjusted_depth = max(0.5, base)
    return adjusted_depth
```

**Mapping Table**:
| Gemini Depth | Adjusted Depth | Movement Speed |
|--------------|----------------|----------------|
| 1 | 0.50 | Very Slow |
| 2 | 0.61 | Slow |
| 3 | 0.83 | Moderate |
| 5 | 1.68 | Normal |
| 7 | 2.86 | Fast |
| 10 | 6.00 | Very Fast |

### Obstacle Detection Pipeline

**Bounding Box Processing**:
```python
# Normalize coordinates to pixel space
if max(obstacle['bounding_box']) <= 1000:
    xmin = int((xmin / 1000.0) * self.image_width)
    ymin = int((ymin / 1000.0) * self.image_height)
    xmax = int((xmax / 1000.0) * self.image_width)
    ymax = int((ymax / 1000.0) * self.image_height)
```

**Visualization System**:
- Red bounding boxes for obstacles
- Green circles for target points
- Labels with obstacle descriptions

### Frame Recording System

**Adaptive Mode**: 3fps continuous recording
**Obstacle Mode**: 10fps high-detail recording

**Recording Structure**:
```
raw_frames/
├── session_YYYYMMDD_HHMMSS/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── flight_YYYYMMDD_HHMMSS/
    ├── frame_000001.jpg
    └── ...
```

## Performance Characteristics

### Latency Analysis

**Adaptive Mode Processing Chain**:
1. Frame Capture: ~50ms
2. Gemini API Call: 2-5 seconds
3. Response Processing: ~100ms
4. Action Execution: 500-3000ms
5. **Total**: 2.65-8.15 seconds per cycle

**Obstacle Mode Processing Chain**:
1. Frame Capture: ~50ms
2. Keepalive Activation: ~10ms
3. Gemini API Call: 3-8 seconds
4. Keepalive Deactivation: ~10ms
5. Response Processing: ~150ms
6. Obstacle Processing: ~50ms
7. Action Execution: 500-3000ms
8. **Total**: 3.77-11.22 seconds per cycle

### Memory Usage Profile

**Baseline Usage**: ~150MB
- Python Runtime: ~50MB
- OpenCV: ~30MB
- Gemini Client: ~40MB
- Application Code: ~30MB

**Additional per Mode**:
- Adaptive Mode: +50MB (depth processing)
- Obstacle Mode: +100MB (obstacle detection + keepalive)

### Scalability Considerations

**Current Limitations**:
- Single drone support only
- Sequential processing (no parallel API calls)
- Limited to Tello hardware constraints

**Future Scalability Options**:
- Multi-drone fleet management
- Parallel API processing
- Enhanced hardware support
- Distributed processing capability

---

*This technical documentation provides comprehensive details about the Unified Tello System architecture. For user-friendly instructions, refer to the main README file.*
