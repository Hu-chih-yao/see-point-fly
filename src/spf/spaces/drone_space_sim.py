import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ActionPoint:
    """Represents a relative movement action in 3D space"""
    dx: float  # Relative movement in each direction
    dy: float
    dz: float
    action_type: str
    screen_x: float = 0.0  # 2D projected coordinates for visualization
    screen_y: float = 0.0
    detected_obstacles: list = None  # Store detected obstacles

    def __str__(self):
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
        """Convert a relative movement action into drone commands"""
        commands = []

        # 1. Calculate yaw angle needed
        target_angle = math.degrees(math.atan2(action.dx, action.dy)) % 360

        # 2. Add yaw command if needed (if there's horizontal movement)
        if abs(action.dx) > 0.01 or abs(action.dy) > 0.01:
            if target_angle > 180:
                commands.append(('yaw_left', int(abs(360 - target_angle) * (7500/360))))
            else:
                commands.append(('yaw_right', int(target_angle * (7500/360))))

        # 3. Add forward movement if needed
        distance_xy = math.sqrt(action.dx**2 + action.dy**2)
        if distance_xy > 0.01:
            commands.append(('pitch_forward', int(distance_xy * 1000)))

        # 4. Add vertical movement if needed
        if abs(action.dz) > 0.01:
            if action.dz > 0:
                commands.append(('increase_throttle', int(abs(action.dz) * 1000)))
            else:
                commands.append(('decrease_throttle', int(abs(action.dz) * 1000)))

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

if __name__ == "__main__":
    test_action_space()
