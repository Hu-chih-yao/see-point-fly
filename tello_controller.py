import cv2
import numpy as np
import time
from pynput.keyboard import Key, Controller, Listener
import google.generativeai as genai
import base64
from dotenv import load_dotenv
import os
import threading
import queue
from collections import deque
from drone_space import DroneActionSpace, ActionPoint
from action_projector import ActionProjector
import json
from datetime import datetime
from djitellopy import Tello  # Import Tello
import logging
import keyboard


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
        self.initialization_delay = 2.0  # Seconds to wait before starting frame grabbing
        
        print(f"Initializing frame provider (waiting {self.initialization_delay}s for camera)...")
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
                    frame = frame_read.frame.copy()  # Make a copy to avoid reference issues
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
                    print(f"Frame update error (retry {retry_count}/{max_retries}): {e}")
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
                cv2.putText(blank, "No frame available", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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
    """
    Controller for the Tello drone, handles communication and commands
    Supports both direct control and action-based control
    """
    
    def __init__(self):
        """Initialize the Tello controller"""
        # Initialize Tello drone
        self.drone = Tello()
        self.is_connected = False
        self.is_flying = False
        self.action_projector = ActionProjector()
        
        # Configure logging
        self.logger = logging.getLogger('TelloController')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Connect to the drone
        try:
            self.drone.connect()
            self.is_connected = True
            self.logger.info("Connected to Tello drone")
            
            # Initialize stream provider
            self.frame_provider = RealtimeFrameProvider(self.drone)
            self.frame_provider.start()
        except Exception as e:
            self.logger.error(f"Failed to connect to Tello: {e}")
            raise
        
        # Initialize command queue and action counter
        self.command_queue = queue.Queue()
        self.action_counter = 0
        self.last_command_time = time.time()
        
        # Initialize keyboard listener for manual control
        self.key_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.key_listener.daemon = True
        self.key_listener.start()
        
        # Manual control state
        self.manual_control_active = False
        self.pressed_keys = set()
        
        # Set output directory for visualizations
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"tello_flight_data/{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.action_projector.set_output_dir(self.output_dir)
        
        # Set control mode and configuration
        self.control_mode = "velocity"  # Options: "velocity", "distance"
        self.config = {
            "distance_scale": 100,  # cm per unit in 3D space
            "velocity_scale": 50,   # Percentage of max speed
            "command_duration": 0.5  # seconds for velocity commands
        }
        
        # Start command processing thread
        self.running = True
        self.command_thread = threading.Thread(target=self._command_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        # Initialize control parameters
        self.action_queue = queue.Queue()
        self.action_history = deque(maxlen=5)  # Keep last 5 actions
        
        # Add manual control flag
        self.manual_control_active = False
        self.manual_key_pressed = None
        
        # Default speed settings
        self.default_speed = 50  # Default speed value
        
        # Map abstract actions to Tello RC control parameters (left_right, forward_backward, up_down, yaw)
        # Format: (left_right, forward_backward, up_down, yaw)
        self.action_map = {
            'increase_throttle': (0, 0, self.default_speed, 0),    # Up
            'decrease_throttle': (0, 0, -self.default_speed, 0),   # Down
            'yaw_left': (0, 0, 0, -self.default_speed),            # Turn left
            'yaw_right': (0, 0, 0, self.default_speed),            # Turn right
            'roll_left': (-self.default_speed, 0, 0, 0),           # Left
            'roll_right': (self.default_speed, 0, 0, 0),           # Right
            'pitch_forward': (0, self.default_speed, 0, 0),        # Forward
            'pitch_back': (0, -self.default_speed, 0, 0),          # Backward
            'land': (0, 0, 0, 0),                                 # Land (placeholder - actual land command handled separately)
            'takeoff': (0, 0, 0, 0)                               # Takeoff (placeholder - actual takeoff handled separately)
        }
        
        # Manual control mapping (key -> (command, duration in ms))
        self.manual_control_map = {
            # Using string representation for special keys
            'Key.up': ('pitch_forward', self.default_speed),       # Forward with up arrow
            'Key.down': ('pitch_back', self.default_speed),        # Backward with down arrow
            'a': ('yaw_left', self.default_speed),                 # Turn left with A
            'd': ('yaw_right', self.default_speed),                # Turn right with D
            'Key.left': ('roll_left', self.default_speed),         # Roll left with left arrow
            'Key.right': ('roll_right', self.default_speed),       # Roll right with right arrow
            'w': ('increase_throttle', self.default_speed),        # Up with W
            's': ('decrease_throttle', self.default_speed),        # Down with S
            'l': ('land', self.default_speed),                      # Land with L
            't': ('takeoff', self.default_speed),                   # Takeoff with T
            'e': (None, self.default_speed)                         # Emergency stop with E
        }
        
        # Opposite actions for oscillation prevention
        self.opposite_actions = {
            'yaw_left': 'yaw_right',
            'yaw_right': 'yaw_left',
            'roll_left': 'roll_right',
            'roll_right': 'roll_left',
            'pitch_forward': 'pitch_back',
            'pitch_back': 'pitch_forward',
            'increase_throttle': 'decrease_throttle',
            'decrease_throttle': 'increase_throttle'
        }
        
        # Initialize Gemini
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        # Get battery level
        battery = self.drone.get_battery()
        print(f"\nBattery level: {battery}%")

        # Initialize action space for command conversion
        self.action_space = DroneActionSpace()
        self.action_projector = ActionProjector()
        
        # Add data collection attributes
        self.data_dir = "tello_training_data"
        self.current_episode = []
        self.episode_count = 0
        os.makedirs(self.data_dir, exist_ok=True)
        
        print("TelloController initialized. Drone connected and ready.")
    
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
                        self.drone.send_rc_control(0, 0, 0, 0)
                        last_action = None
                    continue
                
                if action:
                    self._execute_tello_action(action)
                    last_action = action
                    
            except Exception as e:
                print(f"Tello control error: {e}")
                # Safety: try to stop the drone on error
                try:
                    self.drone.send_rc_control(0, 0, 0, 0)
                except:
                    pass
                
    def _execute_tello_action(self, action_tuple):
        """Execute a single action with duration on Tello"""
        action, duration_ms = action_tuple
        
        # Handle special commands that aren't RC controls
        if action == 'land':
            try:
                print("Landing drone")
                self.drone.land()
                return
            except Exception as e:
                print(f"Landing failed: {e}")
                return
        
        if action == 'takeoff':
            try:
                print("Taking off")
                self.drone.takeoff()
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
                self.drone.send_rc_control(lr, fb, ud, yaw)
                
                # Hold for duration
                time.sleep(duration_ms / 1000.0)
                
                # Record time before stopping
                before_stop_time = time.time()
                
                # Stop movement after duration
                self.drone.send_rc_control(0, 0, 0, 0)
                
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
                self.drone.send_rc_control(0, 0, 0, 0)
    
    def _on_key_press(self, key):
        """Handle manual key press for override"""
        try:
            # Convert Key object to string representation for comparison
            key_str = str(key)
            if hasattr(key, 'char'):
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
                    self.drone.send_rc_control(0, 0, 0, 0)
                elif cmd == 'land':
                    print("MANUAL OVERRIDE: Landing")
                    self.drone.land()
                elif cmd == 'takeoff':
                    print("MANUAL OVERRIDE: Taking off")
                    self.drone.takeoff()
                else:
                    print(f"MANUAL OVERRIDE: {cmd}")
                    lr, fb, ud, yaw = self.action_map.get(cmd, (0, 0, 0, 0))
                    self.drone.send_rc_control(lr, fb, ud, yaw)
                
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
            if hasattr(key, 'char'):
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
                if cmd not in ['land', 'takeoff']:
                    # Stop the movement (only if it's not a one-time action)
                    self.drone.send_rc_control(0, 0, 0, 0)
                
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
            cv2.putText(blank, "Tello camera error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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
    
    def process_spatial_command(self, image: np.ndarray, instruction: str, mode: str = "single") -> str:
        """Process command using spatial understanding with obstacle awareness"""
        try:
            # Increment action counter
            self.action_counter += 1
            action_num = self.action_counter
            
            # Set mode and get actions with obstacle detection
            self.action_projector.set_mode(mode)
            actions = self.action_projector.get_gemini_points(image, instruction)
            
            if not actions:
                return "No valid actions identified"
            
            # Build response text and visualization
            response_text = f"\n=== Action #{action_num} ({mode.upper()} mode) ===\n"
            viz_image = image.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Process waypoints
            for i, action in enumerate(actions):
                # Add action details to response
                response_text += f"\nWaypoint {i+1}/{len(actions)}:"
                response_text += f"\n→ Moving: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})"
                
                # Check for obstacles
                has_obstacles = hasattr(action, 'detected_obstacles') and len(action.detected_obstacles) > 0
                if has_obstacles:
                    num_obstacles = len(action.detected_obstacles)
                    response_text += f"\n⚠️ Detected {num_obstacles} obstacles - proceeding with caution"
                    
                    # Log obstacle information
                    for j, obstacle in enumerate(action.detected_obstacles):
                        label = obstacle.get('label', 'unknown')
                        response_text += f"\n  • Obstacle {j+1}: {label}"
                
                # Draw waypoint on visualization
                cv2.circle(viz_image, 
                          (int(action.screen_x), int(action.screen_y)), 
                          10, (0, 255, 0), -1)
                
                # Add label
                cv2.putText(
                    viz_image,
                    f"{i+1}: ({action.dx:.1f}, {action.dy:.1f}, {action.dz:.1f})",
                    (int(action.screen_x) + 15, int(action.screen_y)),
                    font, 0.7, (255, 255, 255), 2
                )
                
                # Draw obstacles if present
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
                
                # Execute the action
                if self.is_connected and self.is_flying:
                    self.execute_action(action)
                    response_text += f"\n✓ Executing movement"
                else:
                    response_text += f"\n✗ Not flying - action simulated only"
            
            # Save visualization with obstacles and waypoints
            viz_path = f"{self.output_dir}/action_{action_num:03d}.jpg"
            cv2.imwrite(viz_path, viz_image)
            
            # Save action data with obstacles
            action_data = {
                "action_number": action_num,
                "mode": mode,
                "instruction": instruction,
                "actions": []
            }
            
            for action in actions:
                action_info = {
                    "dx": float(action.dx),
                    "dy": float(action.dy),
                    "dz": float(action.dz),
                    "screen_x": int(action.screen_x),
                    "screen_y": int(action.screen_y)
                }
                
                # Add obstacles if present
                if hasattr(action, 'detected_obstacles'):
                    action_info["obstacles"] = action.detected_obstacles
                    
                action_data["actions"].append(action_info)
            
            json_path = f"{self.output_dir}/decision_{action_num:03d}.json"
            with open(json_path, 'w') as f:
                json.dump(action_data, f, indent=2)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing command: {str(e)}"

    def execute_action(self, action):
        """Execute a spatial action with obstacle awareness"""
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

    def _execute_distance_action(self, action):
        """Execute action using distance-based commands"""
        # Get scaling factor from config
        distance_scale = self.config.get("distance_scale", 100)
        
        # Calculate movement magnitudes
        distance_x = int(action.dx * distance_scale)
        distance_y = int(action.dy * distance_scale)
        distance_z = int(action.dz * distance_scale)
        
        if abs(distance_x) > 20 or abs(distance_y) > 20 or abs(distance_z) > 20:
            command = None
            
            # Handle x movement (left/right)
            if abs(distance_x) > 20:
                if distance_x > 0:
                    command = f"right {abs(distance_x)}"
                else:
                    command = f"left {abs(distance_x)}"
                self.add_command(command)
            
            # Handle y movement (forward/backward)
            if abs(distance_y) > 20:
                if distance_y > 0:
                    command = f"forward {abs(distance_y)}"
                else:
                    command = f"back {abs(distance_y)}"
                self.add_command(command)
            
            # Handle z movement (up/down)
            if abs(distance_z) > 20:
                if distance_z > 0:
                    command = f"up {abs(distance_z)}"
                else:
                    command = f"down {abs(distance_z)}"
                self.add_command(command)
                
            return True
        else:
            self.logger.info("Movement too small, skipping")
            return False

    def _execute_velocity_action(self, action):
        """Execute action using velocity-based commands"""
        # Calculate velocities (scale -1 to 1 to -100 to 100)
        velocity_scale = self.config.get("velocity_scale", 50)
        command_duration = self.config.get("command_duration", 0.5)
        
        # Scale velocities by configured factor
        lr_velocity = int(action.dx * velocity_scale)  # left/right velocity
        fb_velocity = int(action.dy * velocity_scale)  # forward/backward velocity
        ud_velocity = int(action.dz * velocity_scale)  # up/down velocity
        yaw_velocity = 0  # yaw velocity
        
        # Has obstacles, consider adjusting velocity
        has_obstacles = hasattr(action, 'detected_obstacles') and len(action.detected_obstacles) > 0
        if has_obstacles:
            # If obstacles detected, reduce speed for safety
            lr_velocity = int(lr_velocity * 0.7)
            fb_velocity = int(fb_velocity * 0.7)
            ud_velocity = int(ud_velocity * 0.7)
            self.logger.info(f"Reduced velocity due to obstacles: {lr_velocity}, {fb_velocity}, {ud_velocity}")
        
        # Only send command if there's meaningful movement
        if abs(lr_velocity) > 10 or abs(fb_velocity) > 10 or abs(ud_velocity) > 10:
            command = f"rc {lr_velocity} {fb_velocity} {ud_velocity} {yaw_velocity}"
            self.add_command(command)
            
            # Schedule stop command after duration
            stop_time = time.time() + command_duration
            stop_command = "rc 0 0 0 0"
            self.add_delayed_command(stop_command, stop_time)
            return True
        else:
            self.logger.info("Movement too small, skipping")
            return False

    def takeoff(self):
        """Takeoff the drone"""
        try:
            self.drone.takeoff()
            print("Tello takeoff")
            time.sleep(2)  # Allow drone to stabilize
        except Exception as e:
            print(f"Takeoff error: {e}")
    
    def land(self):
        """Land the drone"""
        try:
            self.drone.land()
            print("Tello landing")
        except Exception as e:
            print(f"Landing error: {e}")
    
    def stop(self):
        """Stop the drone and cleanup"""
        self.running = False
        
        # Stop the drone movement
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
        except:
            pass
        
        # Land if still flying
        try:
            self.drone.land()
        except:
            pass
            
        # Stop video stream
        try:
            self.drone.streamoff()
        except:
            pass
        
        # Stop frame provider
        if hasattr(self, 'frame_provider'):
            self.frame_provider.stop()
        
        # Stop keyboard listener
        if hasattr(self, 'key_listener') and self.key_listener.is_alive():
            self.key_listener.stop()
            
        # Stop keyboard thread
        if self.command_thread.is_alive():
            self.command_thread.join(timeout=1.0)

        print("TelloController stopped and cleaned up") 