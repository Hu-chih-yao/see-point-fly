# Drone Obstacle Avoidance System

This document provides a comprehensive technical overview of the obstacle avoidance system implemented in the drone_agent_v7 project. The system uses a unique approach combining computer vision, depth estimation, and LLM-based reasoning to identify potential collisions and generate safe flight paths.

## 1. Overview Architecture

The obstacle avoidance system takes a primarily **planning-based** rather than reactive approach. Instead of detecting obstacles and then reacting to them during flight, the system:

1. Detects obstacles and potential flight paths simultaneously
2. Plans safe paths that inherently avoid obstacles
3. Executes planned movements without requiring real-time adjustments

This approach leverages the Gemini multimodal model to understand the visual scene, identify obstacles, and generate optimal waypoints in a single operation.

## 2. High-Level Processing Pipeline

### 2.1 Entry Point

The obstacle avoidance process begins in `TelloIntegration.process_command()` in `tello_integration.py`:

```python
def process_command(self, instruction, mode="single"):
    """Process a high-level command using the action projector"""
    try:
        # Get current frame from drone
        frame = self.get_frame()
        if frame is None:
            return "No video frame available"
        
        # Set mode for action projector
        self.action_projector.set_mode(mode)
        
        # Get actions and visualization
        actions_and_viz = self.action_projector.get_gemini_points(frame, instruction)
        if not actions_and_viz or len(actions_and_viz) != 2:
            return "No valid actions identified"
        
        actions, viz_image = actions_and_viz
        if not actions:
            return "No valid actions identified"
        
        action = actions[0]  # Get first action
        
        # Execute the action
        self.execute_action(action)
        
        return f"Executed action #{action_num}: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})"
            
    except Exception as e:
        self.logger.error(f"Error processing command: {e}")
        return f"Error: {str(e)}"
```

### 2.2 Execution Flow

The overall pipeline follows these steps:

1. Capture video frame from drone camera
2. Process frame through the ActionProjector
3. Generate actions with embedded obstacle awareness
4. Execute actions while monitoring for obstacles
5. Capture new frame and repeat

## 3. Gemini-Based Obstacle Detection

### 3.1 Combined Detection and Planning

The core innovation is using Gemini for simultaneous obstacle detection and path planning in `action_projector.py`:

```python
def _get_single_action(self, image: np.ndarray, instruction: str) -> Tuple[ActionPoint, np.ndarray]:
    """Get single next best action and its visualization"""
    if not self.output_dir:
        raise ValueError("Output directory not set. Call set_output_dir() first.")
        
    try:
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"""You are a drone navigation expert. Looking at this drone camera view:

        Task: {instruction}

        1. Point to the SINGLE best next position for the drone to move.
        2. Identify any obstacles in the path.

        Return in JSON format:
        {{
            "point": [y, x],
            "label": "action description",
            "obstacles": [
                {{"bounding_box": [ymin, xmin, ymax, xmax], "label": "obstacle_description"}}
            ]
        }}

        Requirements:
        - Choose ONE optimal next position
        - Consider immediate obstacles
        - Make incremental progress toward goal
        - Position should be in clear space
        - Identify any obstacles that could block the path
        """

        # Get response from Gemini
        response = self.model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': encoded_image}
        ])
```

### 3.2 Response Processing

The system processes Gemini's response to extract both the waypoint and detected obstacles:

```python
# Parse JSON response
response_data = json.loads(response_text)

# Get waypoint coordinates
y, x = response_data['point']

# Convert normalized coordinates to pixel coordinates
pixel_x = int((x / 1000.0) * self.image_width)
pixel_y = int((y / 1000.0) * self.image_height)

# Project 2D point to 3D
x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y))

# Create ActionPoint
action = ActionPoint(
    dx=x3d, dy=y3d, dz=z3d,
    action_type="move",
    screen_x=pixel_x,
    screen_y=pixel_y
)

# Add obstacles if present
if 'obstacles' in response_data:
    obstacles = []
    for obstacle in response_data['obstacles']:
        if 'bounding_box' in obstacle:
            ymin, xmin, ymax, xmax = obstacle['bounding_box']
            # Convert to pixel coordinates if normalized
            if max(obstacle['bounding_box']) <= 1000:
                xmin = int((xmin / 1000.0) * self.image_width)
                ymin = int((ymin / 1000.0) * self.image_height)
                xmax = int((xmax / 1000.0) * self.image_width)
                ymax = int((ymax / 1000.0) * self.image_height)
            obstacle['bounding_box'] = [ymin, xmin, ymax, xmax]
        obstacles.append(obstacle)
    action.detected_obstacles = obstacles
```

## 4. Waypoint Planning with Obstacle Awareness

### 4.1 3D Space Projection

The system converts 2D image coordinates to 3D movement vectors using perspective projection:

```python
def reverse_project_point(self, point_2d: Tuple[int, int], depth: float = 1) -> Tuple[float, float, float]:
    """Project 2D image point back to 3D space"""
    # Center and normalize coordinates
    x_normalized = (point_2d[0] - self.image_width/2) / (self.image_width/2)
    y_normalized = (self.image_height/2 - point_2d[1]) / (self.image_height/2)
    
    # Adjust depth based on vertical position (closer if lower in image)
    depth_factor = 1.0 + (y_normalized * 0.5)  # Adjust depth based on height
    depth = depth * depth_factor
    
    # Calculate 3D coordinates with optimized depth
    x = depth * x_normalized * np.tan(np.radians(self.fov_horizontal/2))
    z = depth * y_normalized * np.tan(np.radians(self.fov_vertical/2))
    y = depth
    
    return (x, y, z)
```

### 4.2 Obstacle-Aware Path Selection

The Gemini prompt explicitly instructs the model to avoid obstacles when selecting waypoints:

```
Requirements:
- Choose ONE optimal next position
- Consider immediate obstacles  
- Make incremental progress toward goal
- Position should be in clear space
- Identify any obstacles that could block the path
```

This ensures that the waypoint selection is already obstacle-aware before execution begins.

## 5. Visualization and Data Storage

### 5.1 Obstacle Visualization

Detected obstacles are visualized with red bounding boxes and labels:

```python
# Draw obstacles
if hasattr(action, 'detected_obstacles'):
    for obstacle in action.detected_obstacles:
        if 'bounding_box' in obstacle:
            ymin, xmin, ymax, xmax = obstacle['bounding_box']
            # Draw rectangle for obstacle
            cv2.rectangle(viz_image, 
                        (int(xmin), int(ymin)), 
                        (int(xmax), int(ymax)),
                        (0, 0, 255), 2)  # Red color for obstacles
            # Add obstacle label
            cv2.putText(viz_image, obstacle.get('label', 'obstacle'),
                       (int(xmin), int(ymin)-10),
                       font, 0.7,
                       (0, 0, 255), 2)
```

### 5.2 Data Persistence

Obstacle information is stored in decision JSON files for analysis and debugging:

```python
# Save action data
action_data = {
    "action_number": action_num,
    "mode": mode,
    "instruction": instruction,
    "actions": [{
        "dx": float(action.dx),
        "dy": float(action.dy),
        "dz": float(action.dz),
        "screen_x": int(action.screen_x),
        "screen_y": int(action.screen_y)
    }]
}

# Add obstacles if present
if hasattr(action, 'detected_obstacles'):
    action_data["actions"][0]["obstacles"] = action.detected_obstacles

json_path = f"{self.output_dir}/decision_{action_num:03d}.json"
with open(json_path, 'w') as f:
    json.dump(action_data, f, indent=2)
```

## 6. Execution with Obstacle Awareness

### 6.1 Safety Check

Before executing movement, the system checks for obstacles:

```python
def execute_action(self, action):
    """Execute a spatial action based on control mode"""
    if not self.is_connected or not self.is_flying:
        self.logger.warning("Cannot execute action: not flying")
        return False
    
    try:
        # Check for obstacles
        has_obstacles = hasattr(action, 'detected_obstacles') and len(action.detected_obstacles) > 0
        if has_obstacles:
            self.logger.warning(f"Detected {len(action.detected_obstacles)} obstacles - proceeding with caution")
        
        # Execute based on control mode
        if self.control_mode == "distance":
            return self._execute_distance_action(action)
        else:
            return self._execute_velocity_action(action)
    except Exception as e:
        self.logger.error(f"Action execution failed: {e}")
        return False
```

### 6.2 Movement Execution

The system executes movements in two modes:

#### Distance Mode:
```python
def _execute_distance_action(self, action):
    """Execute action using distance-based commands"""
    # Get scaling factor from config
    distance_scale = self.config.get("distance_scale", 100)
    
    # Calculate movement magnitudes
    distance_x = int(action.dx * distance_scale)
    distance_y = int(action.dy * distance_scale)
    distance_z = int(action.dz * distance_scale)
    
    # Calculate rotation angle
    if abs(action.dx) > 0.05 or abs(action.dy) > 0.05:
        # Calculate angle in degrees
        angle = math.degrees(math.atan2(action.dx, action.dy))
        yaw_angle = int(angle)
        if yaw_angle < 0:
            # Convert to clockwise angle
            yaw_angle = 360 + yaw_angle
    else:
        yaw_angle = 0
    
    # ... rest of execution code ...
```

#### Velocity Mode:
```python
def _execute_velocity_action(self, action):
    """Execute action using velocity-based commands"""
    # Calculate velocities (scale -1 to 1 to -100 to 100)
    lr_velocity = int(action.dx * 100)  # left/right velocity
    fb_velocity = int(action.dy * 100)  # forward/backward velocity
    ud_velocity = int(action.dz * 100)  # up/down velocity
    
    # ... rest of execution code ...
```

## 7. Technical Implementation Details

### 7.1 Planning vs. Reactive Avoidance

This system employs a **planning-based** approach to obstacle avoidance rather than a reactive one. Key differences:

- **Planning-Based**: Obstacles are accounted for during waypoint selection
- **Reactive**: Would adjust movement parameters during execution

The code does not show explicit movement parameter adjustments in response to obstacles during execution. Instead, the obstacle avoidance happens primarily at the planning stage.

### 7.2 Integration of Computer Vision and Decision Making

The system integrates computer vision and decision making in a novel way:

1. Traditional systems: 
   - Use separate object detection models 
   - Apply path planning algorithms
   - Combine results with rules-based systems

2. This system:
   - Uses a single multimodal model (Gemini)
   - Performs detection and planning simultaneously
   - Integrates visual understanding with spatial reasoning

### 7.3 Obstacle Data Structure

Obstacles are represented with bounding boxes and labels:

```python
obstacles = [
    {
        "bounding_box": [ymin, xmin, ymax, xmax],
        "label": "obstacle_description"
    },
    # ... more obstacles
]
```

These are attached to the `ActionPoint` objects via the `detected_obstacles` attribute.

## 8. Feedback Loop

The system implements a continuous feedback loop:

1. Capture frame
2. Detect obstacles and plan path
3. Execute movement
4. Capture new frame
5. Repeat

This allows it to adapt to new obstacles or changed environments with each iteration.

## 9. Limitations and Future Improvements

### Current Limitations

1. The system relies entirely on Gemini for obstacle detection without traditional computer vision backup
2. There is no explicit collision avoidance during execution, only during planning
3. The system does not maintain a map of previously detected obstacles

### Potential Improvements

1. Hybrid approach combining traditional object detection with LLM guidance
2. Explicit reactive collision avoidance as a safety backup
3. Obstacle memory across frames to handle partially visible obstacles
4. Dynamic speed adjustment based on proximity to detected obstacles 