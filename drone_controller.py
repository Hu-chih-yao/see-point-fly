import cv2
import numpy as np
import mss
import time
from pynput.keyboard import Key, Controller, Listener
import google.generativeai as genai
import base64
from io import BytesIO
from dotenv import load_dotenv
import os
import threading
import queue
from collections import deque
from drone_space import DroneActionSpace, ActionPoint
from action_projector import ActionProjector
import json
from datetime import datetime
import yaml  # Add this import

class DroneController:
    def __init__(self):
        self.keyboard = Controller()
        self.action_queue = queue.Queue()
        self.running = True
        self.action_history = deque(maxlen=5)  # Keep last 5 actions
        
        # Add manual control flag
        self.manual_control_active = False
        self.manual_key_pressed = None
        
        # Start keyboard control thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_control_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        # Start manual override keyboard listener
        self.key_listener = Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release)
        self.key_listener.daemon = True
        self.key_listener.start()
        
        # Action mapping
        self.action_map = {
            'increase_throttle': 'w',
            'decrease_throttle': 's',
            'yaw_left': 'a',
            'yaw_right': 'd',
            'roll_left': Key.left,
            'roll_right': Key.right,
            'pitch_forward': Key.up,
            'pitch_back': Key.down
        }
        
        # Manual control mapping (key -> (command, duration in ms))
        self.manual_control_map = {
            'i': ('pitch_forward', 70),  # Forward
            'k': ('pitch_back', 70),     # Backward
            'j': ('yaw_left', 70),       # Left
            'l': ('yaw_right', 70),      # Right
            'q': ('roll_left', 70),      # Roll left
            'e': ('roll_right', 70),     # Roll right
            'r': ('increase_throttle', 70),  # Up
            'f': ('decrease_throttle', 70),  # Down
            'x': (None, 0)               # Emergency stop
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
        
        # Initialize action space for command conversion
        self.action_space = DroneActionSpace()
        self.action_projector = ActionProjector()
        
        # Add data collection attributes
        self.data_dir = "drone_training_data"
        self.current_episode = []
        self.episode_count = 0
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _keyboard_control_loop(self):
        """Separate thread for keyboard control"""
        while self.running:
            try:
                action = self.action_queue.get(timeout=0.1)
                if action:
                    self._execute_action(action)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Keyboard control error: {e}")
                
    def _execute_action(self, action_tuple):
        """Execute a single action with duration"""
        action, duration_ms = action_tuple
        
        if action in self.action_map:
            key = self.action_map[action]
            try:
                print(f"Executing {action} for {duration_ms}ms")
                # Actually press the key
                self.keyboard.press(key)
                time.sleep(duration_ms / 1000.0)  # Convert ms to seconds
                self.keyboard.release(key)
                
                # Update drone state
                new_state = self.action_space.update_state(action, duration_ms)
                print(f"New state: {new_state}")
                
                # Small pause between actions
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Keyboard action failed: {e}")
                # Make sure to release key if error occurs
                self.keyboard.release(key)
    
    def _on_key_press(self, key):
        """Handle manual key press for override"""
        try:
            # Convert Key object to string representation if needed
            key_char = key.char if hasattr(key, 'char') else str(key)
            
            # Check if key is in our manual control map
            if key_char.lower() in self.manual_control_map:
                # Set the manual control flag
                self.manual_control_active = True
                self.manual_key_pressed = key_char.lower()
                
                # Clear the action queue to stop AI commands
                self.clear_action_queue()
                
                # Execute manual command
                cmd, duration = self.manual_control_map[key_char.lower()]
                if cmd is None:  # Emergency stop
                    print("EMERGENCY STOP")
                    for action in list(self.action_map.values()):
                        try:
                            self.keyboard.release(action)
                        except:
                            pass
                else:
                    print(f"MANUAL OVERRIDE: {cmd}")
                    action_key = self.action_map.get(cmd)
                    if action_key:
                        self.keyboard.press(action_key)
                
        except AttributeError:
            # Special keys like Shift, etc.
            pass
        except Exception as e:
            print(f"Error in manual control: {e}")
    
    def _on_key_release(self, key):
        """Handle manual key release"""
        try:
            key_char = key.char if hasattr(key, 'char') else str(key)
            
            if key_char.lower() in self.manual_control_map:
                # Release the key
                cmd, _ = self.manual_control_map[key_char.lower()]
                if cmd is not None:
                    action_key = self.action_map.get(cmd)
                    if action_key:
                        self.keyboard.release(action_key)
                
                # Reset manual control if this was the active key
                if self.manual_key_pressed == key_char.lower():
                    self.manual_key_pressed = None
                    # Only deactivate manual mode if no other keys are pressed
                    # This gives a small window to transition between keys
                    if self.manual_key_pressed is None:
                        self.manual_control_active = False
                        print("Returning to AI control")
        
        except AttributeError:
            pass
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
        
    def stop(self):
        """Stop the keyboard control thread"""
        self.running = False
        
        # Stop keyboard listener
        if hasattr(self, 'key_listener') and self.key_listener.is_alive():
            self.key_listener.stop()
            
        # Stop keyboard thread
        if self.keyboard_thread.is_alive():
            self.keyboard_thread.join()
    
    def wait_for_queue_empty(self, timeout=30, debug=False):
        """Wait until action queue is empty or timeout occurs
        
        Args:
            timeout (float): Maximum time to wait in seconds
            debug (bool): Whether to print detailed debug information
            
        Returns:
            bool: True if queue emptied, False if timed out
        """
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
        """Process command using spatial understanding system"""
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
        """Execute a single spatial action"""
        commands = self.action_space.action_to_commands(action)
        
        for cmd, duration in commands:
            if cmd in self.action_map:
                if not quiet:
                    print(f"Executing: {cmd} ({duration}ms)")
                self.execute_action((cmd, duration))
                time.sleep(duration/1000.0)  # Reduced delay

def print_monitor_info():
    """Print information about available monitors for debugging"""
    with mss.mss() as sct:
        for i, monitor in enumerate(sct.monitors):
            print(f"Monitor {i}: {monitor}")

def capture_screen(monitor_index=1):
    """Capture the simulator screen
    
    Args:
        monitor_index: Index of the monitor to capture (1=main monitor, 0=all monitors)
    """
    try:
        with mss.mss() as sct:
            # Get monitor information
            if monitor_index >= len(sct.monitors):
                print(f"Warning: Monitor index {monitor_index} out of range. Using main monitor (1).")
                monitor_index = 1
                
            monitor = sct.monitors[monitor_index]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Print monitor dimensions every 100 captures (commented out by default)
            # import random
            # if random.random() < 0.01:  # 1% chance to print monitor info
            #     print(f"Monitor {monitor_index} dimensions: {img.shape[1]}x{img.shape[0]}")
                
            return img
    except Exception as e:
        print(f"Error capturing screen: {e}")
        # Return a blank image with error message as fallback
        blank = np.zeros((2214, 3420, 3), dtype=np.uint8)
        cv2.putText(blank, "Screen capture error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return blank

def main():
    """Main control loop"""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    drone_controller = DroneController()
    
    try:
        # Get initial command from user
        current_command = input("Enter high-level command (e.g., 'navigate through the center of the crane structure'): ")
        
        print("\nStarting control loop in", config['mode'], "mode...")
        print("Press Ctrl+C to exit")
        
        while True:
            # Capture current view
            frame = capture_screen()
            
            # Process command using spatial understanding
            response = drone_controller.process_spatial_command(
                frame, 
                current_command, 
                mode=config['mode']
            )
            print(f"\nAction Response:\n{response}\n")
            
            # Add configured delay between actions
            time.sleep(config['command_loop_delay'])
            
            # Loop back to get next action from Gemini
            # No asking for completion, just continuous processing
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        drone_controller.stop()
''' Reference code for testing
def test_controller():
    """Test the drone controller with spatial understanding"""
    try:
        # Initialize controller
        controller = DroneController()
        
        # Load test image
        image = cv2.imread('frame_1733321874.11946.jpg')
        if image is None:
            raise ValueError("Could not load test image")
            
        instruction = "navigate through the center of the crane structure while avoiding obstacles"
        
        # Test single action mode
        print("\n=== Testing Single Action Mode ===")
        response = controller.process_spatial_command(image, instruction, mode="single")
        print(response)
        
        # Test waypoint mode
        print("\n=== Testing Waypoint Mode ===")
        response = controller.process_spatial_command(image, instruction, mode="waypoint")
        print(response)
        
    except KeyboardInterrupt:
        print("\nCaught Ctrl+C, exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.stop()
'''
if __name__ == "__main__":
    print("Starting in 3 seconds... Switch to simulator window!")
    time.sleep(3)
    main()