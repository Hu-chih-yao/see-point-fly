import time
import airsim
import threading
import queue
from collections import deque
from .drone_space import AirSimDroneActionSpace
from .action_projector import AirSimActionProjector
from ..base.drone_space import ActionPoint


class AirSimController:
    def __init__(
        self, adaptive_mode=True, image_width=1920, image_height=1080, config=None
    ):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Apply wind settings from config
        if config:
            wind_x = config.get("wind_x", 0.0)
            wind_y = config.get("wind_y", 0.0)
            wind_z = config.get("wind_z", 0.0)
            wind = airsim.Vector3r(wind_x, wind_y, wind_z)
            self.client.simSetWind(wind)
            print(f"Wind set to: X={wind_x}, Y={wind_y}, Z={wind_z} m/s (NED frame)")

        self.action_queue = queue.Queue()
        self.running = True
        self.action_history = deque(maxlen=5)
        self.adaptive_mode = adaptive_mode

        self.image_width = image_width
        self.image_height = image_height

        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        self.action_space = AirSimDroneActionSpace()
        self.action_projector = AirSimActionProjector(
            image_width=self.image_width,
            image_height=self.image_height,
            adaptive_mode=self.adaptive_mode,
            config_path="config_airsim.yaml",
        )

        print(
            f"AirSimController initialized in {'adaptive' if self.adaptive_mode else 'obstacle'} mode."
        )

    def _control_loop(self):
        """Separate thread for drone control"""
        while self.running:
            try:
                action = self.action_queue.get(timeout=0.1)
                if action:
                    self._execute_action(action)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Control error: {e}")

    def _execute_action(self, command_tuple):
        """Execute a single AirSim command"""
        command_type, params = command_tuple

        try:
            if command_type == "rotate_yaw":
                angle = params["angle"]
                yaw_rate = params.get("yaw_rate", 30.0)
                duration = abs(angle) / yaw_rate
                rate = yaw_rate if angle > 0 else -yaw_rate
                print(f"Executing rotate_yaw: {angle:.1f}° at {yaw_rate:.1f}°/s")
                self.client.rotateByYawRateAsync(rate, duration).join()
                self.client.hoverAsync().join()

            elif command_type == "move_velocity_body":
                vx = params["vx"]
                vy = params["vy"]
                vz = params["vz"]
                duration = params["duration"]
                yaw_rate = params.get("yaw_rate", 0)

                print(
                    f"[MOVEMENT] Executing move (body): vx={vx:.2f} m/s, vy={vy:.2f} m/s, vz={vz:.2f} m/s (DOWN+/UP-), yaw_rate={yaw_rate:.1f}°/s, duration={duration:.2f}s"
                )

                # Execute movement command
                yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)

                self.client.moveByVelocityBodyFrameAsync(
                    vx,
                    vy,
                    vz,
                    duration,
                    airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode,
                ).join()

                # Immediately engage hover mode
                self.client.hoverAsync().join()

        except Exception as e:
            print(f"Command execution failed: {e}")
            import traceback

            traceback.print_exc()

    def wait_for_queue_empty(self, timeout=30, debug=False):
        """Wait until action queue is empty or timeout occurs"""
        start_time = time.time()
        if debug:
            print(f"Queue size before waiting: {self.action_queue.qsize()}")

        while not self.action_queue.empty():
            if debug:
                print(f"Queue not empty, remaining items: {self.action_queue.qsize()}")

            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                print("Warning: Timed out waiting for action queue to empty")
                return False
            time.sleep(0.1)

        if debug:
            print(f"Queue emptied after {time.time() - start_time:.2f} seconds")
        return True

    def execute_action(self, command_tuple):
        """Add action to queue"""
        self.action_queue.put(command_tuple)

    def stop(self):
        """Stop the control thread and reset drone"""
        self.running = False
        if self.control_thread.is_alive():
            self.control_thread.join()

        self.client.hoverAsync().join()
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def takeoff(self):
        """Takeoff the drone"""
        print("Taking off...")
        self.client.takeoffAsync().join()
        print("Takeoff complete")

    def process_spatial_command(self, current_frame, instruction: str):
        """Process command using spatial understanding system"""
        try:
            actions = self.action_projector.get_vlm_points(current_frame, instruction)

            if not actions:
                return "No valid actions identified"

            response_text = "\n=== SINGLE ACTION MODE ===\n"

            action = actions[0]
            if action is None:
                return "No valid action"
            response_text += (
                f"\n→ Moving: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})"
            )
            self._execute_spatial_action(action, quiet=True)

            return response_text

        except Exception as e:
            print(f"Error: {e}")
            return "Error processing command"

    def _execute_spatial_action(self, action: ActionPoint, quiet: bool = False):
        """Execute a single spatial action"""
        commands = self.action_space.action_to_commands(action)

        for cmd_type, params in commands:
            if not quiet:
                print(f"Executing: {cmd_type} with {params}")
            self.execute_action((cmd_type, params))
