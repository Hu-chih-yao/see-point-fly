import cv2
import numpy as np
import time
from pynput.keyboard import Listener
import os
import threading
import queue
from collections import deque
from .drone_space import TelloDroneActionSpace
from .action_projector import TelloActionProjector
from ..base.drone_space import ActionPoint
from datetime import datetime
from djitellopy import Tello


class FrameRecorder:
    """
    Records frames from the Tello drone at a specified rate.
    Creates a new folder for each recording session.
    """

    def __init__(self, frame_provider, fps=3, base_dir="raw_frames"):
        self.frame_provider = frame_provider
        self.fps = fps
        self.interval = 1.0 / fps
        self.base_dir = base_dir
        self.running = False
        self.recording_thread = None
        self.frames_saved = 0
        self.session_dir = None

        # Ensure base directory exists
        os.makedirs(base_dir, exist_ok=True)

    def start_recording(self, session_name=None):
        """Start recording frames in a new session folder"""
        if self.running:
            print("Recording already in progress")
            return False

        # Create session directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_name:
            self.session_dir = os.path.join(
                self.base_dir, f"{session_name}_{timestamp}"
            )
        else:
            self.session_dir = os.path.join(self.base_dir, f"session_{timestamp}")

        os.makedirs(self.session_dir, exist_ok=True)

        # Reset counter
        self.frames_saved = 0

        # Start recording thread
        self.running = True
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        print(f"Started frame recording at {self.fps}fps in: {self.session_dir}")
        return True

    def stop_recording(self):
        """Stop the current recording session"""
        if not self.running:
            return False

        self.running = False

        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        print(
            f"Stopped recording. Saved {self.frames_saved} frames to {self.session_dir}"
        )
        return True

    def _recording_loop(self):
        """Main recording loop that runs in a separate thread"""
        last_save_time = time.time()

        while self.running:
            current_time = time.time()
            elapsed = current_time - last_save_time

            # Check if it's time to save a frame
            if elapsed >= self.interval:
                # Capture and save frame
                frame = self.frame_provider.get_frame()
                if frame is not None and frame.size > 0:
                    frame_filename = os.path.join(
                        self.session_dir, f"frame_{self.frames_saved:06d}.jpg"
                    )
                    # Save frame in BGR format for OpenCV
                    cv2.imwrite(frame_filename, frame)
                    self.frames_saved += 1

                # Update last save time (accounting for drift)
                last_save_time = current_time

            # Sleep a small amount to prevent CPU spinning
            # Use a shorter sleep than the interval to maintain timing accuracy
            sleep_time = max(0.005, self.interval / 4)
            time.sleep(sleep_time)


class RealtimeFrameProvider:
    """
    Dedicated provider that continuously updates and provides the latest frame from Tello.
    Ensures that any frame access is getting the absolute most recent camera view.
    """

    def __init__(self, tello):
        self.tello = tello
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.initialization_delay = (
            2.0  # Seconds to wait before starting frame grabbing
        )

        print(
            f"Initializing frame provider (waiting {self.initialization_delay}s for camera)..."
        )
        time.sleep(self.initialization_delay)  # Allow camera time to stabilize

        # Start background thread for frame updates
        self.update_thread = threading.Thread(target=self._update_frame_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def _update_frame_loop(self):
        """Continuously update the latest frame"""
        retry_count = 0
        max_retries = 5
        retry_delay = 0.5

        while self.running:
            try:
                # Get frame read object (avoiding frequent recreation)
                frame_read = self.tello.get_frame_read()
                if frame_read and frame_read.frame is not None:
                    frame = (
                        frame_read.frame.copy()
                    )  # Make a copy to avoid reference issues
                    if frame is not None and frame.size > 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with self.frame_lock:
                            self.latest_frame = frame
                            self.frame_count += 1
                            retry_count = 0  # Reset retry count on success

                # Use a slower update rate to reduce resource contention
                time.sleep(0.05)  # 20fps is plenty for our needs

            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    print(
                        f"Frame update error (retry {retry_count}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay)
                else:
                    print(f"Frame update failed after {max_retries} retries: {e}")
                    # Don't spam logs with errors, wait longer between retries after max is reached
                    time.sleep(1.0)

    def get_frame(self):
        """Get the absolute latest frame"""
        with self.frame_lock:
            if self.latest_frame is None:
                # Return blank frame as fallback
                blank = np.zeros((720, 960, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "No frame available",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
                return blank
            return self.latest_frame.copy()

    def get_frame_count(self):
        """Get the number of frames processed since startup"""
        with self.frame_lock:
            return self.frame_count

    def stop(self):
        """Stop the frame provider"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)


class TelloController:
    def __init__(self, mode="adaptive_mode"):
        self.tello = Tello()  # Create Tello instance
        self.tello.connect()
        self.tello.streamon()

        # Initialize real-time frame provider
        self.frame_provider = RealtimeFrameProvider(self.tello)

        # Store operational mode
        self.operational_mode = mode

        # Initialize frame recorder with mode-specific FPS
        fps = 10 if mode == "obstacle_mode" else 3
        self.frame_recorder = FrameRecorder(self.frame_provider, fps=fps)

        # Initialize control parameters
        self.action_queue = queue.Queue()
        self.running = True
        self.action_history = deque(maxlen=5)  # Keep last 5 actions

        # Add manual control flag
        self.manual_control_active = False
        self.manual_key_pressed = None

        # Default speed settings
        self.default_speed = 100  # Default speed value

        # Start control thread
        self.control_thread = threading.Thread(target=self._tello_control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        # Start keepalive thread (obstacle_mode only)
        if mode == "obstacle_mode":
            self.keepalive_active = True
            self.last_command_time = time.time()
            self.keepalive_interval = 5  # Default interval in seconds
            self.intensive_keepalive = False  # Flag for intensive keepalive mode
            self.keepalive_thread = threading.Thread(target=self._keepalive_loop)
            self.keepalive_thread.daemon = True
            self.keepalive_thread.start()
            print("[KEEPALIVE] Thread started - will prevent automatic landing")
        else:
            self.keepalive_active = False

        # Start manual override keyboard listener
        self.key_listener = Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self.key_listener.daemon = True
        self.key_listener.start()

        # Map abstract actions to Tello RC control parameters (left_right, forward_backward, up_down, yaw)
        # Format: (left_right, forward_backward, up_down, yaw)
        self.action_map = {
            "increase_throttle": (0, 0, self.default_speed, 0),  # Up
            "decrease_throttle": (0, 0, -self.default_speed, 0),  # Down
            "yaw_left": (0, 0, 0, -self.default_speed),  # Turn left
            "yaw_right": (0, 0, 0, self.default_speed),  # Turn right
            "roll_left": (-self.default_speed, 0, 0, 0),  # Left
            "roll_right": (self.default_speed, 0, 0, 0),  # Right
            "pitch_forward": (0, self.default_speed, 0, 0),  # Forward
            "pitch_back": (0, -self.default_speed, 0, 0),  # Backward
            "land": (
                0,
                0,
                0,
                0,
            ),  # Land (placeholder - actual land command handled separately)
            "takeoff": (
                0,
                0,
                0,
                0,
            ),  # Takeoff (placeholder - actual takeoff handled separately)
        }

        # Manual control mapping (key -> (command, duration in ms))
        self.manual_control_map = {
            # Using string representation for special keys
            "Key.up": ("pitch_forward", self.default_speed),  # Forward with up arrow
            "Key.down": ("pitch_back", self.default_speed),  # Backward with down arrow
            "a": ("yaw_left", self.default_speed),  # Turn left with A
            "d": ("yaw_right", self.default_speed),  # Turn right with D
            "Key.left": ("roll_left", self.default_speed),  # Roll left with left arrow
            "Key.right": (
                "roll_right",
                self.default_speed,
            ),  # Roll right with right arrow
            "w": ("increase_throttle", self.default_speed),  # Up with W
            "s": ("decrease_throttle", self.default_speed),  # Down with S
            "l": ("land", self.default_speed),  # Land with L
            "t": ("takeoff", self.default_speed),  # Takeoff with T
            "e": (None, self.default_speed),  # Emergency stop with E
        }

        # Opposite actions for oscillation prevention
        self.opposite_actions = {
            "yaw_left": "yaw_right",
            "yaw_right": "yaw_left",
            "roll_left": "roll_right",
            "roll_right": "roll_left",
            "pitch_forward": "pitch_back",
            "pitch_back": "pitch_forward",
            "increase_throttle": "decrease_throttle",
            "decrease_throttle": "increase_throttle",
        }

        # Get battery level
        battery = self.tello.get_battery()
        print(f"\nBattery level: {battery}%")

        # Initialize action space for command conversion
        self.action_space = TelloDroneActionSpace()
        self.action_projector = TelloActionProjector(
            mode=mode, config_path="config_tello.yaml"
        )

        print(f"TelloController initialized in {mode} mode. Drone connected and ready.")

    def start_frame_recording(self, session_name=None):
        """Start recording frames at 10fps"""
        return self.frame_recorder.start_recording(session_name)

    def stop_frame_recording(self):
        """Stop recording frames"""
        return self.frame_recorder.stop_recording()

    def _tello_control_loop(self):
        """Separate thread for Tello control"""
        last_action = None

        while self.running:
            try:
                # Get next action from queue with timeout
                action = None
                try:
                    action = self.action_queue.get(timeout=0.1)
                except queue.Empty:
                    # If queue is empty and no manual control, stop the drone
                    if not self.manual_control_active and last_action is not None:
                        self.tello.send_rc_control(0, 0, 0, 0)
                        last_action = None
                    continue

                if action:
                    self._execute_tello_action(action)
                    last_action = action

            except Exception as e:
                print(f"Tello control error: {e}")
                # Safety: try to stop the drone on error
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                except:
                    pass

    def _execute_tello_action(self, action_tuple):
        """Execute a single action with duration on Tello"""
        action, duration_ms = action_tuple

        # Handle special commands that aren't RC controls
        if action == "land":
            try:
                print("Landing drone")
                self.tello.land()
                return
            except Exception as e:
                print(f"Landing failed: {e}")
                return

        if action == "takeoff":
            try:
                print("Taking off")
                self.tello.takeoff()
                return
            except Exception as e:
                print(f"Takeoff failed: {e}")
                return

        # Handle regular RC commands
        if action in self.action_map:
            lr, fb, ud, yaw = self.action_map[action]
            try:
                print(f"▶ {action} ({duration_ms}ms)")

                # Record start time
                start_time = time.time()

                # Send RC command to Tello
                self.tello.send_rc_control(lr, fb, ud, yaw)

                # Hold for duration
                time.sleep(duration_ms / 1000.0)

                # Record time before stopping
                before_stop_time = time.time()

                # Stop movement after duration
                self.tello.send_rc_control(0, 0, 0, 0)

                # Record end time
                end_time = time.time()

                # Calculate and Print actual durations
                actual_duration_ms = (before_stop_time - start_time) * 1000
                total_command_time_ms = (end_time - start_time) * 1000
                difference_ms = actual_duration_ms - duration_ms
                print(f"✓ Done: {actual_duration_ms:.0f}ms (Δ{difference_ms:+.1f}ms)")

                # Update drone state (using original action space)
                new_state = self.action_space.update_state(action, duration_ms)
                print(f"New state: {new_state}")

                # Small pause between actions
                time.sleep(0.1)

            except Exception as e:
                print(f"Tello action failed: {e}")
                # Safety: try to stop the drone
                self.tello.send_rc_control(0, 0, 0, 0)

    def _on_key_press(self, key):
        """Handle manual key press for override"""
        try:
            # Convert Key object to string representation for comparison
            key_str = str(key)
            if hasattr(key, "char"):
                key_char = key.char.lower()
            else:
                key_char = None

            # Check if key is in our manual control map (either by char or full key string)
            matches_key = False
            manual_cmd = None

            # Check if key matches any entry in our map
            for map_key, cmd_info in self.manual_control_map.items():
                if (key_char and map_key == key_char) or key_str == map_key:
                    matches_key = True
                    manual_cmd = cmd_info
                    matched_key = map_key
                    break

            if matches_key:
                # Set the manual control flag
                self.manual_control_active = True
                self.manual_key_pressed = matched_key

                # Clear the action queue to stop AI commands
                self.clear_action_queue()

                # Execute manual command
                cmd, duration = manual_cmd
                if cmd is None:  # Emergency stop
                    print("EMERGENCY STOP")
                    self.tello.send_rc_control(0, 0, 0, 0)
                elif cmd == "land":
                    print("MANUAL OVERRIDE: Landing")
                    self.tello.land()
                elif cmd == "takeoff":
                    print("MANUAL OVERRIDE: Taking off")
                    self.tello.takeoff()
                else:
                    print(f"MANUAL OVERRIDE: {cmd}")
                    lr, fb, ud, yaw = self.action_map.get(cmd, (0, 0, 0, 0))
                    self.tello.send_rc_control(lr, fb, ud, yaw)

        except AttributeError:
            # Special keys handling
            pass
        except Exception as e:
            print(f"Error in manual control: {e}")

    def _on_key_release(self, key):
        """Handle manual key release"""
        try:
            # Convert Key object to string representation for comparison
            key_str = str(key)
            if hasattr(key, "char"):
                key_char = key.char.lower()
            else:
                key_char = None

            # Check if the released key was the active manual control key
            matches_key = False
            for map_key in self.manual_control_map.keys():
                if (key_char and map_key == key_char) or key_str == map_key:
                    matches_key = True
                    matched_key = map_key
                    break

            if matches_key:
                # Get the command associated with this key
                cmd, _ = self.manual_control_map[matched_key]

                # For land and takeoff, we don't need to stop any movement
                if cmd not in ["land", "takeoff"]:
                    # Stop the movement (only if it's not a one-time action)
                    self.tello.send_rc_control(0, 0, 0, 0)

                # Reset manual control if this was the active key
                if self.manual_key_pressed == matched_key:
                    self.manual_key_pressed = None
                    # Only deactivate manual mode if no other keys are pressed
                    if self.manual_key_pressed is None:
                        self.manual_control_active = False
                        print("Returning to AI control")

        except Exception as e:
            print(f"Error in manual control release: {e}")

    def clear_action_queue(self):
        """Clear all pending actions from the queue"""
        try:
            count = 0
            while not self.action_queue.empty():
                self.action_queue.get_nowait()
                self.action_queue.task_done()
                count += 1
            if count > 0:
                print(f"Cleared {count} AI commands from queue")
        except Exception as e:
            print(f"Error clearing queue: {e}")

    def is_manual_control_active(self):
        """Check if manual control is currently active"""
        return self.manual_control_active

    def execute_action(self, action_tuple):
        """Add action to queue"""
        self.action_queue.put(action_tuple)

    def capture_frame(self):
        """Capture the absolute latest frame from Tello camera"""
        try:
            # Use the frame provider to get the latest frame
            return self.frame_provider.get_frame()
        except Exception as e:
            print(f"Error capturing Tello frame: {e}")
            # Return a blank image with error message as fallback
            blank = np.zeros((720, 960, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Tello camera error",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            return blank

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
            time.sleep(0.1)  # Short sleep to prevent CPU spinning

        if debug:
            print(f"Queue emptied after {time.time() - start_time:.2f} seconds")
        return True

    def process_spatial_command(
        self, current_frame, instruction: str, mode: str = "single"
    ):
        """Process command using spatial understanding system with mode-specific handling"""
        try:
            # Mode-specific processing for obstacle_mode
            if self.operational_mode == "obstacle_mode":
                # Record start time for API call
                api_start_time = time.time()
                print(
                    f"[KEEPALIVE] API call starting at {time.strftime('%H:%M:%S')} - keepalive protection active"
                )

                # Start intensive keepalive before API call
                self.start_intensive_keepalive()

                # Pass controller reference to action projector for keepalive control
                actions = self.action_projector.get_vlm_points(
                    current_frame, instruction, tello_controller=self
                )

                # Return to normal keepalive after API call
                self.stop_intensive_keepalive()

                # Report API call duration
                api_duration = time.time() - api_start_time
                print(f"[KEEPALIVE] API call completed in {api_duration:.2f} seconds")
            else:
                # Adaptive mode - standard processing
                actions = self.action_projector.get_vlm_points(
                    current_frame, instruction
                )

            if not actions:
                return "No valid actions identified"

            response_text = f"\\n=== {self.operational_mode.upper()} ===\\n"

            # Execute single action
            action = actions[0]
            if action is None:
                return "No valid action"

            if self.operational_mode == "obstacle_mode":
                print("\\n action in process_spatial_command part(obstacle mode):")
                print("/n", actions)

            response_text += (
                f"\\n→ Moving: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})"
            )
            self._execute_spatial_action(action, quiet=True)

            return response_text

        except Exception as e:
            print(f"Error: {e}")
            return "Error processing command"

    def _execute_spatial_action(self, action: ActionPoint, quiet: bool = False):
        """Execute a single spatial action - adapted for Tello"""
        commands = self.action_space.action_to_commands(action)

        for cmd, duration in commands:
            if cmd in self.action_map:
                if not quiet:
                    print(f"Executing: {cmd} ({duration}ms)")
                self.execute_action((cmd, duration))
                time.sleep(duration / 1000.0)  # Reduced delay

    def takeoff(self):
        """Takeoff the drone"""
        try:
            self.tello.takeoff()
            print("Tello takeoff")
            time.sleep(2)  # Allow drone to stabilize
        except Exception as e:
            print(f"Takeoff error: {e}")

    def land(self):
        """Land the drone"""
        try:
            self.tello.land()
            print("Tello landing")
        except Exception as e:
            print(f"Landing error: {e}")

    def _keepalive_loop(self):
        """Send keepalive commands to prevent the drone from auto-landing after 15 seconds (obstacle_mode only)"""
        last_status_time = 0
        status_interval = 30  # Check and print status every 30 seconds

        while self.running and self.keepalive_active:
            try:
                current_time = time.time()

                # Only send keepalive when flying and not manually controlled
                if self.tello.is_flying and not self.manual_control_active:
                    # Send keepalive command
                    self.tello.send_keepalive()
                    if self.intensive_keepalive:
                        print(
                            f"[KEEPALIVE-INTENSIVE] Signal sent at {time.strftime('%H:%M:%S')}"
                        )
                    else:
                        print(f"[KEEPALIVE] Signal sent at {time.strftime('%H:%M:%S')}")

                # Periodically print status information
                if current_time - last_status_time > status_interval:
                    self.check_drone_status()
                    last_status_time = current_time

            except Exception as e:
                print(f"[KEEPALIVE] Error: {e}")

            # Use shorter interval during intensive mode
            if self.intensive_keepalive:
                time.sleep(1)  # Send every 1 second during API calls
            else:
                time.sleep(self.keepalive_interval)  # Normal interval

    def check_drone_status(self):
        """Check and print comprehensive drone status (obstacle_mode only)"""
        try:
            bat = self.tello.get_battery()
            temp = self.tello.get_temperature()
            height = self.tello.get_height()
            flight_time = self.tello.get_flight_time()

            print(
                f"[TELLO STATUS] Battery: {bat}%, Temp: {temp}°C, Height: {height}cm, Flight time: {flight_time}s"
            )

            if bat < 20:
                print("[TELLO WARNING] Battery level critical!")

            return bat, temp, height, flight_time
        except Exception as e:
            print(f"[TELLO ERROR] Status check failed: {e}")
            return None

    def start_intensive_keepalive(self):
        """Start sending keepalive signals more frequently during API calls (obstacle_mode only)"""
        if self.operational_mode == "obstacle_mode":
            self.intensive_keepalive = True
            print("[KEEPALIVE] Starting intensive mode (1-second interval)")

    def stop_intensive_keepalive(self):
        """Return to normal keepalive frequency (obstacle_mode only)"""
        if self.operational_mode == "obstacle_mode":
            self.intensive_keepalive = False
            print("[KEEPALIVE] Returning to normal mode")

    def stop(self):
        """Stop the drone and cleanup"""
        # Stop keepalive thread (obstacle_mode only)
        if self.operational_mode == "obstacle_mode":
            self.keepalive_active = False
        self.running = False

        # Stop frame recording if active
        if hasattr(self, "frame_recorder"):
            self.frame_recorder.stop_recording()

        # Stop the drone movement
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except:
            pass

        # Land if still flying
        try:
            self.tello.land()
        except:
            pass

        # Stop video stream
        try:
            self.tello.streamoff()
        except:
            pass

        # Stop frame provider
        if hasattr(self, "frame_provider"):
            self.frame_provider.stop()

        # Stop keyboard listener
        if hasattr(self, "key_listener") and self.key_listener.is_alive():
            self.key_listener.stop()

        # Stop threads
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        if (
            self.operational_mode == "obstacle_mode"
            and hasattr(self, "keepalive_thread")
            and self.keepalive_thread.is_alive()
        ):
            self.keepalive_thread.join(timeout=1.0)

        print(f"TelloController ({self.operational_mode}) stopped and cleaned up")
