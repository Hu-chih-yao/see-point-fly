import time
from pynput.keyboard import Key, Controller
import os
import threading
import queue
from collections import deque
from .drone_space import SimDroneActionSpace
from .action_projector import SimActionProjector
from ..base.drone_space import ActionPoint


class SimController:
    def __init__(self, adaptive_mode=True, screen_width=1920, screen_height=1080):
        self.keyboard = Controller()
        self.action_queue = queue.Queue()
        self.running = True
        self.action_history = deque(maxlen=5)  # Keep last 5 actions
        self.adaptive_mode = adaptive_mode  # Store adaptive mode setting

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Start keyboard control thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_control_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        # Action mapping
        self.action_map = {
            "increase_throttle": "w",
            "decrease_throttle": "s",
            "yaw_left": "a",
            "yaw_right": "d",
            "roll_left": Key.left,
            "roll_right": Key.right,
            "pitch_forward": Key.up,
            "pitch_back": Key.down,
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

        # Initialize action space for command conversion
        self.action_space = SimDroneActionSpace()
        self.action_projector = SimActionProjector(
            image_width=self.screen_width,
            image_height=self.screen_height,
            adaptive_mode=self.adaptive_mode,
            config_path="config_sim.yaml",
        )

        print(
            f"SimController initialized in {'adaptive' if self.adaptive_mode else 'obstacle'} mode."
        )

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

    def execute_action(self, action_tuple):
        """Add action to queue"""
        self.action_queue.put(action_tuple)

    def stop(self):
        """Stop the keyboard control thread"""
        self.running = False
        if self.keyboard_thread.is_alive():
            self.keyboard_thread.join()

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

        for cmd, duration in commands:
            if cmd in self.action_map:
                if not quiet:
                    print(f"Executing: {cmd} ({duration}ms)")
                self.execute_action((cmd, duration))
                time.sleep(duration / 1000.0)  # Reduced delay
