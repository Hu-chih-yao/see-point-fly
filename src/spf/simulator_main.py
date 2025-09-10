#!/usr/bin/env python3
"""
Simulator main module for SPF (See, Point, Fly)
Uses screen capture, depth estimation, and LLM-based command processing
"""

import os
import time
import cv2
import numpy as np
import mss

from .controllers.sim_controller import SimController
from .projectors.action_projector_sim import ActionProjectorSim

def print_monitor_info():
    """Print information about available monitors"""
    with mss.mss() as sct:
        for i, monitor in enumerate(sct.monitors):
            if i == 0:
                print(f"Monitor {i} (All): {monitor}")
            else:
                print(f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")

def detect_screen_dimensions(monitor_index=1):
    """Detect screen dimensions for specified monitor

    Args:
        monitor_index: Index of the monitor to detect (1=main monitor, 0=all monitors)

    Returns:
        tuple: (screen_width, screen_height)
    """
    try:
        with mss.mss() as sct:
            # Use monitor 1 (main monitor) to get dimensions
            monitor = sct.monitors[monitor_index] if len(sct.monitors) > monitor_index else sct.monitors[0]
            screen_width = monitor['width']
            screen_height = monitor['height']
            print(f"Detected screen dimensions: {screen_width}x{screen_height}")
            return screen_width, screen_height
    except Exception as e:
        print(f"Error detecting screen dimensions: {e}")
        # Return default dimensions as fallback
        return 1920, 1080

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

            # Print monitor dimensions every 100 captures (commented out by default)
            # import random
            # if random.random() < 0.01:  # 1% chance to print monitor info
            #     print(f"Monitor {monitor_index} dimensions: {img.shape[1]}x{img.shape[0]}")

            return img
    except Exception as e:
        print(f"Error capturing screen: {e}")
        # Return a blank image with error message as fallback
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(blank, "Screen capture error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return blank

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
        # Load config for debug mode
        try:
            import yaml
            with open('config_sim.yaml', 'r') as f:
                config = yaml.safe_load(f)
                adaptive_mode = config.get('adaptive_mode', True)
        except Exception as e:
            print(f"Config loading failed, using default adaptive mode: {e}")
            adaptive_mode = True

        # Detect screen dimensions
        monitor_index = getattr(args, 'monitor', 1)
        screen_width, screen_height = detect_screen_dimensions(monitor_index)

        # Create coordinate system visualization
        action_projector = ActionProjectorSim(
            image_width=screen_width,
            image_height=screen_height,
            adaptive_mode=adaptive_mode
        )

        print(f"Monitor {monitor_index} dimensions: {screen_width}x{screen_height}")
        print(f"ActionProjectorSim dimensions: {action_projector.image_width}x{action_projector.image_height}")

        if screen_width != action_projector.image_width or screen_height != action_projector.image_height:
            print("\nWARNING: Monitor dimensions don't match ActionProjectorSim dimensions!")
            print("This may cause incorrect coordinate projections.")
            print(f"Consider updating ActionProjectorSim to use {screen_width}x{screen_height}")

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

        # Load config for test mode
        try:
            import yaml
            with open('config_sim.yaml', 'r') as f:
                config = yaml.safe_load(f)
                adaptive_mode = config.get('adaptive_mode', True)
        except Exception as e:
            print(f"Config loading failed, using default adaptive mode: {e}")
            adaptive_mode = True

        # Get image dimensions for test mode
        image_height, image_width = test_image.shape[:2]
        print(f"Using test image dimensions: {image_width}x{image_height}")

        # Create controller
        controller = SimController(
            adaptive_mode=adaptive_mode,
            screen_width=image_width,
            screen_height=image_height
        )

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
            adaptive_mode = config.get('adaptive_mode', True)
            print(f"Adaptive Mode: {adaptive_mode}")
            print(f"Command loop delay: {config.get('command_loop_delay', 0)}s")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        config = {'command_loop_delay': 0, 'adaptive_mode': True}
        adaptive_mode = True

    # Detect screen dimensions
    screen_width, screen_height = detect_screen_dimensions(monitor_index)

    # Create controller with adaptive mode configuration
    sim_controller = SimController(
        adaptive_mode=adaptive_mode,
        screen_width=screen_width,
        screen_height=screen_height
    )

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
                sim_controller.wait_for_queue_empty(debug=True)
                print("Action queue empty, processing new frame...")
            else:
                sim_controller.wait_for_queue_empty()

            # Capture current view from specified monitor
            frame = capture_screen(monitor_index=monitor_index)

            if frame is None:
                print("Error: Failed to capture screen")
                time.sleep(1)
                continue

            # Process command
            response = sim_controller.process_spatial_command(
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
        if 'sim_controller' in locals():
            sim_controller.stop()

    return 0
