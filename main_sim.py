#!/usr/bin/env python3
"""
Main drone control application
Uses screen capture, depth estimation, and LLM-based command processing
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
import argparse
from drone_controller_sim import DroneController, capture_screen, print_monitor_info
from action_projector_sim import ActionProjector

# Import fixed screen capture
try:
    from tools.capture.fixed_capture import capture_screen_fixed as capture_screen
    print("Using resolution-fixed screen capture")
except ImportError:
    from drone_controller_sim import capture_screen
    print("Using default screen capture")

def main():
    """Main entrypoint with improved startup and diagnostics"""
    parser = argparse.ArgumentParser(description='Drone Spatial Navigation System')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor index (1=primary monitor)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--test', action='store_true', help='Run test with static image')
    parser.add_argument('--info', action='store_true', help='Display monitor information and exit')
    args = parser.parse_args()
    
    # Special case: just print monitor info and exit
    if args.info:
        print("\n=== AVAILABLE MONITORS ===")
        print_monitor_info()
        return 0
    
    # Debug mode
    if args.debug:
        print("\n=== DEBUG MODE ===")
        # Create coordinate system visualization
        action_projector = ActionProjector()
        
        # Get current screen resolution
        with mss.mss() as sct:
            monitor = sct.monitors[args.monitor]
            print(f"Monitor {args.monitor} dimensions: {monitor['width']}x{monitor['height']}")
            print(f"ActionProjector dimensions: {action_projector.image_width}x{action_projector.image_height}")
            
            if monitor['width'] != action_projector.image_width or monitor['height'] != action_projector.image_height:
                print("\nWARNING: Monitor dimensions don't match ActionProjector dimensions!")
                print("This may cause incorrect coordinate projections.")
                print(f"Consider updating ActionProjector to use {monitor['width']}x{monitor['height']}")
        
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
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        # Create controller
        controller = DroneController()
        
        # Test instruction
        instruction = "navigate through the crane structure safely"
        
        # Process with test image
        response = controller.process_spatial_command(test_image, instruction, mode="waypoint")
        print(f"\nAction Response:\n{response}\n")
        
        return 0
    
    # Normal operation
    print("\n=== STARTING DRONE SPATIAL NAVIGATION ===")
    print(f"Using monitor {args.monitor}")
    
    
    # Load config
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"Mode: {config['mode']}")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration (single mode)")
        config = {'mode': 'single', 'command_loop_delay': 0}
    
    # Create controller
    drone_controller = DroneController()
    
    try:
        # Get initial command from user
        current_command = input("\nEnter high-level command (e.g., 'navigate through the center of the crane structure'): ")
        
        print("Starting in 3 seconds... Switch to simulator window!")
        time.sleep(3)
        print("\nStarting control loop...")
        print("Press Ctrl+C to exit")
        
        while True:
            # Capture current view from specified monitor
            frame = capture_screen(monitor_index=args.monitor)
            
            if frame is None:
                print("Error: Failed to capture screen")
                time.sleep(1)
                continue
                
            # Process command
            response = drone_controller.process_spatial_command(
                frame, 
                current_command, 
                mode=config['mode']
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

if __name__ == "__main__":
    sys.exit(main()) 