import cv2
import numpy as np
import mss
import time
from pynput.keyboard import Key, Controller
import google.generativeai as genai
import base64
from io import BytesIO
from dotenv import load_dotenv
import os
import threading
import queue
from collections import deque
from drone_space import DroneActionSpace, ActionPoint
from action_projector_sim import ActionProjector
import json
from datetime import datetime
import yaml  # Add this import

class DroneController:
    def __init__(self):
        self.keyboard = Controller()
        self.action_queue = queue.Queue()
        self.running = True
        self.action_history = deque(maxlen=5)  # Keep last 5 actions
        
        # Start keyboard control thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_control_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
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
            model_name="gemini-2.0-flash-exp",
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
    
    def _should_avoid_action(self, proposed_action):
        """Check if action would cause oscillation"""
        if not self.action_history:
            return False
            
        opposite = self.opposite_actions.get(proposed_action)
        if len(self.action_history) >= 2:
            last_two = list(self.action_history)[-2:]
            if last_two == [proposed_action, opposite] or last_two == [opposite, proposed_action]:
                print(f"Avoiding oscillation between {proposed_action} and {opposite}")
                return True
        return False
    
    def process_drone_command(self, current_frame, original_command, last_execution=None):
        """Process drone command using Gemini's spatial understanding"""
        try:
            # Get points from Gemini
            actions = self.action_projector.get_gemini_points(current_frame, original_command)
            
            if not actions:
                return "No valid actions identified"
            
            # Execute waypoints sequentially
            response_text = "Executing waypoint sequence:\n"
            
            for i, action in enumerate(actions, 1):
                response_text += f"\n=== Waypoint {i}/{len(actions)} ===\n"
                response_text += f"Target: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})\n"
                
                # Convert 3D action to drone commands
                commands = self.action_space.action_to_commands(action)
                
                # Execute commands for this waypoint
                for cmd, duration in commands:
                    if cmd in self.action_map:
                        print(f"Executing {cmd} for {duration}ms")
                        self.execute_action((cmd, duration))
                        response_text += f"  {cmd}: {duration}ms\n"
                    else:
                        print(f"Warning: Invalid command {cmd}")
                
                # Add small delay between waypoints
                time.sleep(0.2)  # 200ms pause between waypoints
                
                # Optionally check if waypoint reached before continuing
                if i < len(actions):
                    response_text += "Waypoint reached, moving to next...\n"
            
            return response_text
            
        except Exception as e:
            print(f"Error in process_drone_command: {e}")
            return "Error processing command"
    
    def execute_action(self, action_tuple):
        """Add action to queue"""
        self.action_queue.put(action_tuple)
        
    def stop(self):
        """Stop the keyboard control thread"""
        self.running = False
        if self.keyboard_thread.is_alive():
            self.keyboard_thread.join()

    def collect_training_sample(self, frame, action, duration_ms, command):
        """Collect a single training sample"""
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(self.data_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Store sample data
        sample = {
            "image_path": image_path,
            "action": action,
            "duration_ms": duration_ms,
            "command": command,
            "timestamp": timestamp
        }
        self.current_episode.append(sample)
    
    def save_episode(self):
        """Save the current episode to disk"""
        if not self.current_episode:
            return
            
        episode_path = os.path.join(self.data_dir, f"episode_{self.episode_count}.json")
        with open(episode_path, 'w') as f:
            json.dump(self.current_episode, f, indent=2)
            
        self.current_episode = []
        self.episode_count += 1

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

def collect_training_data():
    drone_controller = DroneController()
    
    try:
        for episode_type in EXAMPLE_EPISODES:
            print(f"\nCollecting data for: {episode_type['task']}")
            
            for variation in range(episode_type['variations']):
                print(f"\nVariation {variation + 1}/{episode_type['variations']}")
                
                for instruction in episode_type['instructions']:
                    print(f"\nInstruction: {instruction}")
                    drone_controller.current_command = instruction
                    
                    while True:
                        # Execute actions until instruction complete
                        frame = capture_screen()
                        response = drone_controller.process_drone_command(frame, instruction)
                        
                        complete = input("Instruction complete? (y/n): ")
                        if complete.lower() == 'y':
                            drone_controller.save_episode()
                            break
                        
                        time.sleep(0.1)  # Control loop rate
                
                # Reset drone position for next variation
                input("Reset drone position and press Enter to continue...")
                
    except KeyboardInterrupt:
        print("\nData collection interrupted")
    finally:
        drone_controller.save_episode()
        drone_controller.stop()

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

if __name__ == "__main__":
    print("Starting in 3 seconds... Switch to simulator window!")
    time.sleep(3)
    main()