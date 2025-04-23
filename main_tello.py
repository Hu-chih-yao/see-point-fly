#!/usr/bin/env python3
"""
Main Tello drone control application
Uses Tello camera feed, depth estimation, and LLM-based command processing
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
import argparse
import yaml
from tello_controller import TelloController
from datetime import datetime

def save_frame_to_directory(frame, directory, prefix="frame"):
    """
    Save a frame to the specified directory with a timestamp
    
    Args:
        frame: The frame to save
        directory: Directory to save the frame in
        prefix: Prefix for the filename
        
    Returns:
        Path to the saved frame
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
    
    # Create filename
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    
    # Save frame
    try:
        cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return filepath
    except Exception as e:
        print(f"Error saving frame to {filepath}: {e}")
        return None

def wait_for_camera_ready(tello_controller, max_attempts=15, delay=1.0, save_frames=True):
    """
    Wait until the Tello camera provides valid frames
    
    Args:
        tello_controller: TelloController instance
        max_attempts: Maximum number of frame capture attempts
        delay: Delay between attempts in seconds
        save_frames: Whether to save frames for debugging
        
    Returns:
        bool: True if camera is ready, False if failed after max attempts
        last_good_frame: The last valid frame that passed the check
    """
    print("\nWaiting for camera to initialize...")
    last_good_frame = None
    
    # Give frame provider time to start capturing frames
    time.sleep(2.0)
    
    # Create directory for debug frames if needed
    if save_frames:
        debug_dir = "tello_debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
    
    for attempt in range(1, max_attempts + 1):
        print(f"Checking camera (attempt {attempt}/{max_attempts})...")
        frame = tello_controller.capture_frame()
        
        # Save each attempt for debugging
        if save_frames and frame is not None:
            debug_path = f"tello_debug_frames/attempt_{attempt}.jpg"
            try:
                cv2.imwrite(debug_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error saving debug frame: {e}")
        
        # Skip checking blank frames
        if frame is None or np.sum(frame) == 0:
            print(f"Received blank frame, retrying in {delay} seconds...")
            time.sleep(delay)
            continue
        
        # Check if frame is valid and not the error placeholder
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Check if the frame has actual content (non-zero standard deviation)
            std_dev = np.std(gray)
            print(f"Frame standard deviation: {std_dev:.2f}")
            
            if std_dev > 10:  # Real camera frames should have variation
                print("Camera ready!")
                
                # Save the good frame with timestamp
                if save_frames:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    good_frame_path = f"tello_debug_frames/camera_ready_{timestamp}.jpg"
                    cv2.imwrite(good_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"Saved ready frame to {good_frame_path}")
                
                last_good_frame = frame
                return True, last_good_frame
        except Exception as e:
            print(f"Error checking frame: {e}")
        
        print(f"Camera not ready yet, waiting {delay} seconds...")
        time.sleep(delay)
    
    print("Warning: Camera initialization timed out. Video stream may not be working properly.")
    return False, last_good_frame

def main():
    """Main entrypoint for Tello drone control"""
    parser = argparse.ArgumentParser(description='Tello Drone Spatial Navigation System')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional logging')
    parser.add_argument('--test', action='store_true', help='Use test mode with static images instead of live feed')
    parser.add_argument('--skip-camera-check', action='store_true', help='Skip camera initialization check')
    parser.add_argument('--record', action='store_true', help='Record frames continuously at 10fps')
    parser.add_argument('--record-session', type=str, help='Name for the recording session (optional)')
    args = parser.parse_args()
    
    # Print welcome banner
    print("\n=== STARTING TELLO DRONE SPATIAL NAVIGATION ===")
    
    # Load config
    try:
        with open('config_tello.yaml', 'r') as f:
            config = yaml.safe_load(f)
            print(f"Mode: {config['mode']}")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration (single mode)")
        config = {'mode': 'single', 'command_loop_delay': 0}
    
    # Test mode (using static image)
    if args.test:
        print("\n=== TEST MODE WITH STATIC IMAGE ===")
        test_image_path = 'drone_training_data/test_frame.jpg'
        
        if not os.path.exists(test_image_path):
            print(f"Error: Test image '{test_image_path}' not found")
            return 1
            
        # Load test image
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        # Create controller
        controller = TelloController()
        
        # Test instruction
        instruction = "navigate forward and slightly to the right to avoid obstacles"
        
        # Process with test image
        response = controller.process_spatial_command(test_image, instruction, mode="waypoint")
        print(f"\nAction Response:\n{response}\n")
        
        controller.stop()
        return 0
    
    # Create controller
    try:
        print("\nConnecting to Tello drone...")
        tello_controller = TelloController()
    except Exception as e:
        print(f"Failed to connect to Tello: {e}")
        return 1
    
    try:
        # Wait for camera to be ready (unless skip flag is provided)
        last_good_frame = None
        if not args.skip_camera_check:
            # Try a few times to initialize camera
            camera_ready = False
            for init_attempt in range(3):
                print(f"\nCamera initialization attempt {init_attempt+1}/3")
                camera_ready, last_good_frame = wait_for_camera_ready(tello_controller)
                if camera_ready:
                    break
                # Short wait between attempts
                print("Retrying camera initialization...")
                time.sleep(2.0)
                
            if not camera_ready and args.debug:
                print("Continuing despite camera initialization issues (debug mode)...")
            elif not camera_ready:
                print("Camera not ready after multiple attempts. Try restarting the Tello or use --skip-camera-check to bypass this check.")
                return 1
        
        # Start frame recording if enabled
        if args.record:
            session_name = args.record_session if args.record_session else "flight"
            tello_controller.start_frame_recording(session_name)
            print(f"Started continuous frame recording at 10fps with session name: {session_name}")
        
        # Get initial command from user
        current_command = input("\nEnter high-level command (e.g., 'navigate through the center of the room'): ")
        
        print("\nStarting control loop...")
        print("Press Ctrl+C to exit")
        print("\nMANUAL OVERRIDE CONTROLS:")
        print("  ↑/↓ (Arrow keys): Forward/Backward")
        print("  A/D: Turn left/right")
        print("  ←/→ (Arrow keys): Roll left/right")
        print("  W/S: Up/Down")
        print("  T: Takeoff")
        print("  L: Land")
        print("  E: Emergency stop (stop all movement)")
        print("\nAI control will resume when no override keys are pressed")

        print("\nStarting in 4 seconds... Prepare for takeoff!")
        time.sleep(5)
        
        # Take off
        tello_controller.takeoff()
        
        # Create directory for storing frames sent to Gemini
        gemini_frames_dir = "Tello_frame_capture"
        os.makedirs(gemini_frames_dir, exist_ok=True)
        
        # Initialize frame counter
        frame_count = 0
        
        while True:
            # Capture current view from Tello camera - now always fresh thanks to RealtimeFrameProvider
            frame = tello_controller.capture_frame()
            
            if frame is None:
                print("Error: Failed to capture frame")
                time.sleep(1)
                continue
            
            # Display frame if debug is enabled
            if args.debug:
                cv2.imshow("Tello View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            
            # Check if manual control is active
            if tello_controller.is_manual_control_active():
                # Skip AI processing during manual control
                if args.debug:
                    print("Manual control active, skipping AI processing")
                time.sleep(0.1)  # Small delay to prevent CPU spin
                continue
            
            # Wait for previous actions to complete before processing new frame
            if args.debug:
                print("Waiting for previous actions to complete...")
                tello_controller.wait_for_queue_empty(debug=True)
                print("Action queue empty, processing new frame...")
            else:
                tello_controller.wait_for_queue_empty()
            
            # Save the frame that will be sent to Gemini
            frame_count += 1
            frame_path = save_frame_to_directory(
                frame, 
                gemini_frames_dir, 
                prefix=f"gemini_frame_{frame_count}"
            )
            if frame_path:
                print(f"Saved frame to Gemini: {os.path.basename(frame_path)}")
                
            # Process command
            response = tello_controller.process_spatial_command(
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
        # Close any open OpenCV windows
        cv2.destroyAllWindows()
        
        if 'tello_controller' in locals():
            print("\nLanding drone and cleaning up...")
            tello_controller.stop()
            
    return 0

if __name__ == "__main__":
    sys.exit(main()) 