import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ActionPoint:
    """Represents a relative movement action in 3D space"""
    dx: float  # Relative movement in each direction
    dy: float
    dz: float
    action_type: str
    screen_x: float = 0.0  # 2D projected coordinates for visualization
    screen_y: float = 0.0
    detected_obstacles: Optional[list] = None  # Store detected obstacles
    adaptive_depth: Optional[float] = None  # Adjusted depth for adaptive movement
    vlm_depth: Optional[float] = None  # Original depth from VLM (1-10)

    def __str__(self):
        if self.adaptive_depth is not None:
            return f"Action({self.action_type}): Move({self.dx:.1f}, {self.dy:.1f}, {self.dz:.1f}), Adaptive Depth: {self.adaptive_depth:.2f}"
        else:
            return f"Action({self.action_type}): Move({self.dx:.1f}, {self.dy:.1f}, {self.dz:.1f})"

class DroneActionSpaceSim:
    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples
        self.movement_unit = 1.0  # 1 unit = 1000ms of movement
        self.camera_fov = 90.0  # degrees
        self.max_movement = 2.0  # maximum movement in any direction

        # Add state tracking
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, z]
        self.current_yaw = 0.0  # degrees

    def sample_actions(self) -> List[ActionPoint]:
        """Sample possible relative movements from current position (0,0,0)"""
        actions = []

        # Sample points in a hemisphere in front of the drone
        for _ in range(self.n_samples):
            # Random spherical coordinates
            distance = np.random.uniform(0.5, self.max_movement)
            azimuth = np.random.uniform(-self.camera_fov/2, self.camera_fov/2)
            elevation = np.random.uniform(-self.camera_fov/4, self.camera_fov/4)

            # Convert to relative Cartesian movements
            dx = distance * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth))
            dy = distance * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth))
            dz = distance * math.sin(math.radians(elevation))

            action = ActionPoint(dx, dy, dz, "move")
            actions.append(action)

        return actions

    def action_to_commands(self, action: ActionPoint) -> List[Tuple[str, int]]:
        """Convert a relative movement action into drone commands with adaptive timing"""
        commands = []

        # Determine timing multiplier based on adaptive depth
        if hasattr(action, 'adaptive_depth') and action.adaptive_depth is not None:
            depth_factor = action.adaptive_depth

            if depth_factor == 0:
                # Special case: depth 0 means no movement (object too close)
                print("[ADAPTIVE TIMING] Depth 0 detected - No movement commands will be generated")
                return []  # Return empty command list for no movement
            else:
                # Use adaptive depth for timing (higher depth = faster movement)
                time_multiplier = depth_factor
                print(f"[ADAPTIVE TIMING] Using depth factor {depth_factor:.2f} for movement timing")
        else:
            # Default timing for non-adaptive mode
            time_multiplier = 1.0

        # Base timing values
        base_rotate_time = 7500  # time to rotate 360 degrees
        base_move_time = 1000    # time to move 1 unit

        # Apply adaptive timing
        rotate_time = int(base_rotate_time / time_multiplier)
        move_time = int(base_move_time / time_multiplier)

        # 1. Calculate yaw angle needed
        target_angle = math.degrees(math.atan2(action.dx, action.dy)) % 360

        # 2. Add yaw command if needed (if there's horizontal movement)
        if abs(action.dx) > 0.01 or abs(action.dy) > 0.01:
            if target_angle > 180:
                yaw_duration = int(abs(360 - target_angle) * (rotate_time/360))
                commands.append(('yaw_left', yaw_duration))
            else:
                yaw_duration = int(target_angle * (rotate_time/360))
                commands.append(('yaw_right', yaw_duration))

        # 3. Add forward movement if needed
        distance_xy = math.sqrt(action.dx**2 + action.dy**2)
        if distance_xy > 0.01:
            forward_duration = int(distance_xy * move_time)
            commands.append(('pitch_forward', forward_duration))

        # 4. Add vertical movement if needed
        if abs(action.dz) > 0.01:
            vertical_duration = int(abs(action.dz) * move_time)
            if action.dz > 0:
                commands.append(('increase_throttle', vertical_duration))
            else:
                commands.append(('decrease_throttle', vertical_duration))

        return commands

    def update_state(self, action: str, duration_ms: float) -> dict:
        """Update drone state based on executed action and duration"""
        # Convert duration to seconds
        duration = duration_ms / 1000.0

        # Movement rates (units per second)
        move_rate = 1.0
        yaw_rate = 51.4  # degrees per second (360/7.0)
        vertical_rate = 1.0

        # Update state based on action
        if action == 'yaw_left':
            self.current_yaw = (self.current_yaw - yaw_rate * duration) % 360
        elif action == 'yaw_right':
            self.current_yaw = (self.current_yaw + yaw_rate * duration) % 360
        elif action == 'pitch_forward':
            # Move in direction of current yaw
            yaw_rad = math.radians(self.current_yaw)
            self.current_position[0] += move_rate * duration * math.sin(yaw_rad)
            self.current_position[1] += move_rate * duration * math.cos(yaw_rad)
        elif action == 'pitch_back':
            # Move opposite to current yaw
            yaw_rad = math.radians(self.current_yaw)
            self.current_position[0] -= move_rate * duration * math.sin(yaw_rad)
            self.current_position[1] -= move_rate * duration * math.cos(yaw_rad)
        elif action == 'increase_throttle':
            self.current_position[2] += vertical_rate * duration
        elif action == 'decrease_throttle':
            self.current_position[2] -= vertical_rate * duration

        # Return current state
        return {
            'position': self.current_position.copy(),
            'yaw': self.current_yaw,
            'last_action': action,
            'duration': duration_ms
        }
