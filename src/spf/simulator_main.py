#!/usr/bin/env python3
"""
Simulator main module for SPF (See, Point, Fly)
Uses screen capture, depth estimation, and LLM-based command processing
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
import mss
from pathlib import Path

# We're now inside the spf package, so imports work directly
from .controllers.sim_controller import SimController
from .projectors.action_projector_sim import ActionProjectorSim

# Import fixed screen capture
try:
    from tools.capture.fixed_capture import capture_screen_fixed as capture_screen
    print("Using resolution-fixed screen capture")
except ImportError:
    try:
        from .controllers.sim_controller import capture_screen
        print("Using default screen capture")
    except ImportError:
        # Fallback implementation
        def capture_screen(monitor_index=1):
            with mss.mss() as sct:
                monitor = sct.monitors[monitor_index]
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                return frame
        print("Using fallback screen capture")

def print_monitor_info():
    """Print information about available monitors"""
    with mss.mss() as sct:
        for i, monitor in enumerate(sct.monitors):
            if i == 0:
                print(f"Monitor {i} (All): {monitor}")
            else:
                print(f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")

def main(args):
    """Main entrypoint with improved startup and diagnostics"""

    # Special case: just print monitor info and exit
    if hasattr(args, 'info') and args.info:
        print("\n=== AVAILABLE MONITORS ===")
        print_monitor_info()
        return 0

    # Debug mode
    if args.debug:
        print("\n=== DEBUG MODE ===")
        # Create coordinate system visualization
        action_projector = ActionProjectorSim()

        # Get current screen resolution
        with mss.mss() as sct:
            monitor = sct.monitors[getattr(args, 'monitor', 1)]
            print(f"Monitor {getattr(args, 'monitor', 1)} dimensions: {monitor['width']}x{monitor['height']}")
            print(f"ActionProjectorSim dimensions: {action_projector.image_width}x{action_projector.image_height}")

            if monitor['width'] != action_projector.image_width or monitor['height'] != action_projector.image_height:
                print("\nWARNING: Monitor dimensions don't match ActionProjectorSim dimensions!")
                print("This may cause incorrect coordinate projections.")
                print(f"Consider updating ActionProjectorSim to use {monitor['width']}x{monitor['height']}")

        # Create visualization
        debug_image = action_projector.visualize_coordinate_system()

        # Display and save
        cv2.imshow("Coordinate System", debug_image)
        cv2.imwrite("coordinate_system_debug.jpg", debug_image)
        print("\nPress any key to exit debug mode...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    # Test mode (using static image)
    if args.test:
        print("\n=== TEST MODE WITH STATIC IMAGE ===")
        test_image_path = 'frame_1733321874.11946.jpg'

        if not os.path.exists(test_image_path):
            print(f"Error: Test image '{test_image_path}' not found")
            return 1

        # Load test image
        test_image = cv2.imread(test_image_path)

        # Create controller
        controller = SimController()

        # Test instruction
        instruction = "navigate through the crane structure safely"

        # Process with test image
        response = controller.process_spatial_command(test_image, instruction)
        print(f"\nAction Response:\n{response}\n")

        return 0

    # Normal operation
    print("\n=== STARTING DRONE SPATIAL NAVIGATION (SIMULATOR) ===")
    monitor_index = getattr(args, 'monitor', 1)
    print(f"Using monitor {monitor_index}")

    # Load config
    try:
        import yaml
        with open('config_sim.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"Command loop delay: {config.get('command_loop_delay', 0)}s")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        config = {'command_loop_delay': 0}

    # Create controller
    drone_controller = SimController()

    try:
        # Get initial command from user
        current_command = input("\nEnter high-level command (e.g., 'navigate through the center of the crane structure'): ")

        print("Starting in 3 seconds... Switch to simulator window!")
        time.sleep(3)
        print("\nStarting control loop...")
        print("Press Ctrl+C to exit")

        while True:
            # Wait for previous actions to complete before processing new frame
            if args.debug:
                print("Waiting for previous actions to complete...")
                drone_controller.wait_for_queue_empty(debug=True)
                print("Action queue empty, processing new frame...")
            else:
                drone_controller.wait_for_queue_empty()

            # Capture current view from specified monitor
            frame = capture_screen(monitor_index=monitor_index)

            if frame is None:
                print("Error: Failed to capture screen")
                time.sleep(1)
                continue

            # Process command
            response = drone_controller.process_spatial_command(
                frame,
                current_command
            )
            print(f"\nAction Response:\n{response}\n")

            # Add delay between actions
            time.sleep(config['command_loop_delay'])

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'drone_controller' in locals():
            drone_controller.stop()

    return 0
