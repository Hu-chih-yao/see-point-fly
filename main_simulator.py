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
from drone_controller import DroneController, capture_screen, print_monitor_info
from action_projector import ActionProjector

# Import fixed screen capture
try:
    from tools.capture.fixed_capture import capture_screen_fixed as capture_screen
    print("Using resolution-fixed screen capture")
except ImportError:
    from drone_controller import capture_screen
    print("Using default screen capture")

def main():
    """Main entrypoint with improved startup and diagnostics"""
    parser = argparse.ArgumentParser(description='Drone Spatial Navigation System')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor index (1=primary monitor)')
    parser.add_argument('--info', action='store_true', help='Display monitor information and exit')
    args = parser.parse_args()
    
    # Special case: just print monitor info and exit
    if args.info:
        print("\n=== AVAILABLE MONITORS ===")
        print_monitor_info()
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
        
        print("\nStarting control loop...")
        print("Press Ctrl+C to exit")
        print("\nMANUAL OVERRIDE CONTROLS:")
        print("  W/S: Forward/Backward")
        print("  A/D: Turn left/right")
        print("  Q/E: Roll left/right")
        print("  R/F: Up/Down")
        print("  X: Emergency stop (release all keys)")
        print("\nAI control will resume when no override keys are pressed")

        print("\nStarting in 3 seconds... Switch to simulator window!")
        time.sleep(3)
        
        while True:
            # Capture current view from specified monitor
            frame = capture_screen(monitor_index=args.monitor)
            
            if frame is None:
                print("Error: Failed to capture screen")
                time.sleep(1)
                continue
            
            # Check if manual control is active
            if drone_controller.is_manual_control_active():
                # Skip AI processing during manual control
                if args.debug:
                    print("Manual control active, skipping AI processing")
                time.sleep(0.1)  # Small delay to prevent CPU spin
                continue
            
            # Wait for previous actions to complete before processing new frame
            # This prevents action queue buildup and ensures we're acting on current state
            if args.debug:
                print("Waiting for previous actions to complete...")
                drone_controller.wait_for_queue_empty(debug=True)
                print("Action queue empty, processing new frame...")
            else:
                drone_controller.wait_for_queue_empty()
                
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