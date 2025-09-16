import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..base.drone_space import DroneActionSpace, ActionPoint

class TelloDroneActionSpace(DroneActionSpace):
    def __init__(self, n_samples: int = 8):
        super().__init__(n_samples)
        #self.rotate_time = 3750 #time to rotate 360 degree (7500->50, 4166->90, 3750->100)
        self.rotate_time = 3400 #time to rotate 360 degree (7500->50, 4166->90, 3750->100)
        self.move_time = 500 #time to move 1 unit (1000->50, 555->90, 500->100)

    def action_to_commands(self, action: ActionPoint) -> List[Tuple[str, int]]:
        """Convert a relative movement action into drone commands"""
        commands = []

        # Check if this is a yaw-only action (object too close)
        if hasattr(action, 'yaw_only') and action.yaw_only:
            print("[SAFETY] Object too close - YAW ONLY mode activated")

            # Only generate yaw commands, no forward/backward or up/down movement
            target_angle = math.degrees(math.atan2(action.dx, action.dy)) % 360

            if abs(action.dx) > 0.01 or abs(action.dy) > 0.01:
                if target_angle > 180:
                    yaw_duration = int(abs(360 - target_angle) * (self.rotate_time/360))
                    commands.append(('yaw_left', yaw_duration))
                    print(f"[YAW ONLY] Yaw left {abs(360 - target_angle):.1f}° ({yaw_duration}ms)")
                else:
                    yaw_duration = int(target_angle * (self.rotate_time/360))
                    commands.append(('yaw_right', yaw_duration))
                    print(f"[YAW ONLY] Yaw right {target_angle:.1f}° ({yaw_duration}ms)")
            else:
                print("[YAW ONLY] No significant horizontal movement, no yaw needed")

            # Skip pitch_forward, increase_throttle, decrease_throttle commands
            print("[YAW ONLY] Skipping forward/backward and up/down movements for safety")
            return commands

        # Normal operation - generate all movement commands
        # 1. Calculate yaw angle needed
        target_angle = math.degrees(math.atan2(action.dx, action.dy)) % 360

        # 2. Add yaw command if needed (if there's horizontal movement)
        if abs(action.dx) > 0.01 or abs(action.dy) > 0.01:
            if target_angle > 180:
                commands.append(('yaw_left', int(abs(360 - target_angle) * (self.rotate_time/360))))
            else:
                commands.append(('yaw_right', int(target_angle * (self.rotate_time/360))))

        # 3. Add forward movement if needed
        distance_xy = math.sqrt(action.dx**2 + action.dy**2)
        if distance_xy > 0.01:
            commands.append(('pitch_forward', int(distance_xy * self.move_time)))

        # 4. Add vertical movement if needed
        if abs(action.dz) > 0.01:
            if action.dz > 0:
                commands.append(('increase_throttle', int(abs(action.dz) * self.move_time)))
            else:
                commands.append(('decrease_throttle', int(abs(action.dz) * self.move_time)))

        return commands
