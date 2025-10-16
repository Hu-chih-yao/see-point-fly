import math
import numpy as np
from typing import List, Tuple
from ..base.drone_space import DroneActionSpace, ActionPoint
import yaml
import os


class AirSimDroneActionSpace(DroneActionSpace):
    def __init__(self, n_samples: int = 8, config_path: str = "config_airsim.yaml"):
        super().__init__(n_samples)

        # Load configuration
        config = self._load_config(config_path)
        self.base_velocity = config.get("base_velocity", 2.0)
        self.base_yaw_rate = config.get("base_yaw_rate", 30.0)
        self.min_command_duration = config.get("min_command_duration", 2.0)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

    def sample_actions(self) -> List[ActionPoint]:
        """Sample possible relative movements from current position (0,0,0)"""
        actions = []

        for _ in range(self.n_samples):
            distance = np.random.uniform(0.5, self.max_movement)
            azimuth = np.random.uniform(-self.camera_fov / 2, self.camera_fov / 2)
            elevation = np.random.uniform(-self.camera_fov / 4, self.camera_fov / 4)

            dx = (
                distance
                * math.cos(math.radians(elevation))
                * math.sin(math.radians(azimuth))
            )
            dy = (
                distance
                * math.cos(math.radians(elevation))
                * math.cos(math.radians(azimuth))
            )
            dz = distance * math.sin(math.radians(elevation))

            action = ActionPoint(dx, dy, dz, "move")
            actions.append(action)

        return actions

    def action_to_commands(self, action: ActionPoint) -> List[Tuple[str, dict]]:
        """Convert a relative movement action into AirSim API commands

        Strategy: Rotate first to face target, then move forward
        """
        commands = []

        if hasattr(action, "adaptive_depth") and action.adaptive_depth is not None:
            depth_factor = action.adaptive_depth

            if depth_factor == 0:
                print(
                    "[ADAPTIVE TIMING] Depth 0 detected - No movement commands will be generated"
                )
                return []
            else:
                velocity_multiplier = depth_factor
                print(
                    f"[ADAPTIVE TIMING] Using depth factor {depth_factor:.2f} for movement velocity"
                )
        else:
            velocity_multiplier = 1.0

        velocity_scale = self.base_velocity * velocity_multiplier
        base_yaw_rate = self.base_yaw_rate * velocity_multiplier

        total_distance = math.sqrt(action.dx**2 + action.dy**2 + action.dz**2)

        if total_distance < 0.01:
            return []

        # Step 1: Calculate yaw angle needed to face the target
        distance_xy = math.sqrt(action.dx**2 + action.dy**2)

        if distance_xy > 0.01:
            # Calculate target angle from dx, dy
            target_angle = math.degrees(math.atan2(action.dx, action.dy))

            # Normalize to -180 to 180 range
            if target_angle > 180:
                target_angle -= 360
            elif target_angle < -180:
                target_angle += 360

            print(
                f"[YAW DEBUG] dx={action.dx:.3f}, dy={action.dy:.3f}, target_angle={target_angle:.1f}°"
            )

            # Add rotation command if angle is significant
            if abs(target_angle) > 10:
                commands.append(
                    ("rotate_yaw", {"angle": target_angle, "yaw_rate": base_yaw_rate})
                )
                print(f"[YAW DEBUG] Adding rotation command: {target_angle:.1f}°")
            else:
                print(
                    f"[YAW DEBUG] Target angle {target_angle:.1f}° is within threshold, no yaw"
                )

        # Step 2: Move forward after rotation (or if no rotation needed)
        # Use full velocity for forward movement
        vx = velocity_scale
        vy = 0  # No sideways movement

        # Calculate vertical velocity to maintain correct angle
        if distance_xy > 0.01:
            vz = (-action.dz / distance_xy) * velocity_scale
            duration = max(distance_xy / velocity_scale, self.min_command_duration)
        else:
            vz = 0
            duration = self.min_command_duration

        print(
            f"[MOVEMENT DEBUG] action.dz={action.dz:.3f}, vx={vx:.2f}, vz={vz:.2f}, duration={duration:.2f}s"
        )

        commands.append(
            (
                "move_velocity_body",
                {
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                    "duration": duration,
                    "yaw_rate": 0,  # No yaw during movement, already rotated
                },
            )
        )

        return commands
