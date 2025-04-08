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


class TelloController:
    def __init__(self):
        self.tello = Tello()  # Create Tello instance
        self.tello.connect()
        self.tello.streamon()
        
        # Initialize control parameters
        self.action_queue = queue.Queue()
        self.running = True
        self.action_history = deque(maxlen=5)  # Keep last 5 actions
        
        # Add manual control flag
        self.manual_control_active = False
        self.manual_key_pressed = None
        
        # Default speed settings
        self.default_speed = 40  # Default speed value
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._tello_control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        # Start manual override keyboard listener
        self.key_listener = Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release)
        self.key_listener.daemon = True
        self.key_listener.start()
        
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
        # Updated key mapping according to requirements
        self.manual_control_map = {
            # Using string representation for special keys
            'Key.up': ('pitch_forward', 50),       # Forward with up arrow
            'Key.down': ('pitch_back', 50),        # Backward with down arrow
            'a': ('yaw_left', 50),                 # Turn left with A
            'd': ('yaw_right', 50),                # Turn right with D
            'Key.left': ('roll_left', 50),         # Roll left with left arrow
            'Key.right': ('roll_right', 50),       # Roll right with right arrow
            'w': ('increase_throttle', 50),        # Up with W
            's': ('decrease_throttle', 50),        # Down with S
            'l': ('land', 0),                      # Land with L
            't': ('takeoff', 0),                   # Takeoff with T
            'e': (None, 0)                         # Emergency stop with E
        }
        
        # Opposite actions for oscillation prevention (same as DroneController)
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
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )

        # Get battery level
        battery = self.tello.get_battery()
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
        if action == 'land':
            try:
                print("Landing drone")
                self.tello.land()
                return
            except Exception as e:
                print(f"Landing failed: {e}")
                return
        
        if action == 'takeoff':
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
                print(f"Executing {action} for {duration_ms}ms")
                # Send RC command to Tello
                self.tello.send_rc_control(lr, fb, ud, yaw)
                # Hold for duration
                time.sleep(duration_ms / 1000.0)
                # Stop movement after duration
                self.tello.send_rc_control(0, 0, 0, 0)
                
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
                    self.tello.send_rc_control(0, 0, 0, 0)
                elif cmd == 'land':
                    print("MANUAL OVERRIDE: Landing")
                    self.tello.land()
                elif cmd == 'takeoff':
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
        """Capture frame from Tello camera"""
        try:
            frame = self.tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
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
    
    def process_spatial_command(self, current_frame, instruction: str, mode: str = "waypoint"):
        """Process command using spatial understanding system - same as DroneController"""
        try:
            # Set mode and get actions
            self.action_projector.set_mode(mode)
            actions = self.action_projector.get_gemini_points(current_frame, instruction)
            
            if not actions:
                return "No valid actions identified"
            
            response_text = f"\n=== {mode.upper()} MODE ===\n"
            
            if mode == "waypoint":
                for i, action in enumerate(actions, 1):
                    response_text += f"\nWaypoint {i}/{len(actions)}:"
                    response_text += f"\n→ Moving: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})"
                    self._execute_spatial_action(action, quiet=True)
            else:
                action = actions[0]
                if action is None:
                    return "No valid action"
                response_text += f"\n→ Moving: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})"
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
                time.sleep(duration/1000.0)  # Reduced delay
    
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
    
    def stop(self):
        """Stop the drone and cleanup"""
        self.running = False
        
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
        
        # Stop keyboard listener
        if hasattr(self, 'key_listener') and self.key_listener.is_alive():
            self.key_listener.stop()
            
        # Stop keyboard thread
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        print("TelloController stopped and cleaned up") 