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
import pyperclip  # For clipboard access

def create_control_panel(command_text="", show_message="", cursor_pos=None):
    """Create a simple control panel with command input area"""
    # Create a white image for the panel
    panel = 255 * np.ones((500, 600, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(panel, "Tello Drone Control Panel", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add command input area
    cv2.rectangle(panel, (20, 50), (580, 100), (200, 200, 200), -1)  # Gray background
    cv2.rectangle(panel, (20, 50), (580, 100), (0, 0, 0), 1)  # Black border
    
    # Show command text with cursor
    display_text = "Command: " + command_text
    if cursor_pos is None:
        display_text += "|"  # Add cursor at end if no position specified
    
    cv2.putText(panel, display_text, (25, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Manual controls section
    cv2.putText(panel, "MANUAL OVERRIDE CONTROLS:", (30, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # List all manual controls
    controls = [
        "Up/Down (Arrow keys): Forward/Backward",
        "A/D: Turn left/right",
        "Left/Right (Arrow keys): Roll left/right",
        "W/S: Up/Down",
        "T: Takeoff",
        "L: Land",
        "E: Emergency stop"
    ]
    
    y_pos = 160
    for control in controls:
        cv2.putText(panel, control, (40, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
    
    # System status
    cv2.putText(panel, "System status:", (30, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Show status message if provided
    if show_message:
        cv2.putText(panel, show_message, (40, 430), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # Instructions
    cv2.putText(panel, "Press ENTER to submit command", (30, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
    cv2.putText(panel, "Press BACKSPACE to delete, Ctrl+V to paste", (30, 480), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
    
    return panel

def handle_panel_key(key, command_text, ctrl_pressed=False):
    """
    Handle key presses for the control panel text input
    
    Args:
        key: The key code from cv2.waitKey()
        command_text: Current command text
        ctrl_pressed: Whether Ctrl key is currently pressed
        
    Returns:
        Updated command text or None if Enter was pressed
    """
    # Check for backspace (different key codes on different systems)
    if key == 8 or key == 127:  # Backspace or Delete key
        return command_text[:-1] if command_text else ""
    
    # Check for paste operation (Ctrl+V)
    elif ctrl_pressed and (key == ord('v') or key == ord('V')):
        try:
            # Get clipboard content
            clipboard_text = pyperclip.paste()
            if clipboard_text:
                return command_text + clipboard_text
        except Exception as e:
            print(f"Paste error: {e}")
        return command_text
    
    # Check for Enter key
    elif key == 13 or key == 10:  # Enter key (CR or LF)
        return None  # Signal to submit command
    
    # Check for regular text input (printable ASCII)
    elif 32 <= key <= 126:  # Printable ASCII characters
        return command_text + chr(key)
        
    # No change for other keys
    return command_text

def wait_for_camera_ready(tello_controller, max_attempts=10, delay=0.5, save_frames=True):
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
        
        # Check if frame is valid and not the error placeholder
        if frame is not None and np.sum(frame) > 0:
            # Make sure it's not just the error image by checking if error text is present
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Check if the frame has actual content (non-zero standard deviation)
            if np.std(gray) > 10:  # Real camera frames should have variation
                print("Camera ready!")
                
                # Display the good frame if available
                cv2.imshow("Camera Ready Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1000)  # Show for 1 second
                
                # Save the good frame with timestamp
                if save_frames:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    good_frame_path = f"tello_debug_frames/camera_ready_{timestamp}.jpg"
                    cv2.imwrite(good_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"Saved ready frame to {good_frame_path}")
                
                last_good_frame = frame
                return True, last_good_frame
                
        print(f"Camera not ready yet, waiting {delay} seconds...")
        time.sleep(delay)
    
    print("Warning: Camera initialization timed out. Video stream may not be working properly.")
    return False, last_good_frame

def frame_consumer_thread(tello_controller, stop_event):
    """Thread function to continuously consume frames from Tello"""
    print("Starting frame consumer thread")
    while not stop_event.is_set():
        try:
            # Just capture and discard frames to keep the buffer from overflowing
            frame = tello_controller.capture_frame()
            time.sleep(0.03)  # ~30fps - adjust as needed
        except Exception as e:
            print(f"Error in frame consumer: {e}")
            time.sleep(0.1)
    print("Frame consumer thread stopped")

def main():
    """Main entrypoint for Tello drone control"""
    # Check for pyperclip installation
    try:
        import pyperclip
    except ImportError:
        print("Warning: pyperclip not installed. Paste functionality will be disabled.")
        print("Install with: pip install pyperclip")
        
    parser = argparse.ArgumentParser(description='Tello Drone Spatial Navigation System')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional logging')
    parser.add_argument('--test', action='store_true', help='Use test mode with static images instead of live feed')
    parser.add_argument('--skip-camera-check', action='store_true', help='Skip camera initialization check')
    parser.add_argument('--show-frames', action='store_true', help='Show frames during frame consumption')
    parser.add_argument('--gui', action='store_true', help='Use GUI input panel', default=True)
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI input panel')
    args = parser.parse_args()
    
    # Allow --no-gui to override --gui
    if args.no_gui:
        args.gui = False
    
    # Print welcome banner
    print("\n=== STARTING TELLO DRONE SPATIAL NAVIGATION ===")
    
    # Load config
    try:
        with open('config.yaml', 'r') as f:
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
    tello_controller = TelloController()
    
    try:
        # Wait for camera to be ready (unless skip flag is provided)
        last_good_frame = None
        if not args.skip_camera_check:
            camera_ready, last_good_frame = wait_for_camera_ready(tello_controller)
            if not camera_ready and args.debug:
                print("Continuing despite camera initialization issues (debug mode)...")
            elif not camera_ready:
                print("Camera not ready. Try restarting the Tello or use --skip-camera-check to bypass this check.")
                return 1
        
        # Start a thread to consume frames while waiting for user input
        frame_consumer_stop = threading.Event()
        frame_consumer = threading.Thread(
            target=frame_consumer_thread, 
            args=(tello_controller, frame_consumer_stop),
            daemon=True
        )
        frame_consumer.start()
        
        # Get high-level command from user
        current_command = ""
        
        if args.gui:
            # GUI mode with control panel
            cv2.namedWindow("Tello Control Panel", cv2.WINDOW_AUTOSIZE)
            
            command_submitted = False
            status_message = "Waiting for command..."
            
            # Show battery information if available
            try:
                battery = tello_controller.tello.get_battery()
                status_message = f"Battery: {battery}% - Waiting for command..."
            except:
                pass
            
            # Track control key state
            ctrl_pressed = False
            
            while not command_submitted:
                # Show control panel
                panel = create_control_panel(current_command, status_message)
                cv2.imshow("Tello Control Panel", panel)
                
                # Wait for key press with short timeout to keep UI responsive
                key = cv2.waitKey(100) & 0xFF
                
                # Skip if no key was pressed
                if key == 255:
                    continue
                
                # Check for control key pressing and releasing
                if key == ord('\x11'):  # Ctrl key (DC1)
                    ctrl_pressed = True
                    continue
                elif key == ord('\x13'):  # Ctrl released (DC3)
                    ctrl_pressed = False
                    continue
                
                # Handle text input
                new_command = handle_panel_key(key, current_command, ctrl_pressed)
                
                # Check if command was submitted (Enter key)
                if new_command is None:
                    if current_command.strip():  # Don't accept empty commands
                        command_submitted = True
                        print(f"Command submitted: {current_command}")
                    else:
                        status_message = "Please enter a command first!"
                else:
                    current_command = new_command
                    
                # This is a direct key press/release that's handled by the panel
                # It doesn't interfere with the global keyboard listener in TelloController
                # The manual control still works even while this UI is active
        else:
            # Traditional console input
            current_command = input("\nEnter high-level command (e.g., 'navigate through the center of the room'): ")
        
        # Stop the frame consumer thread after input received
        frame_consumer_stop.set()
        frame_consumer.join(timeout=1.0)
        
        print("\nStarting control loop...")
        print("Press Ctrl+C to exit")
        print("\nMANUAL OVERRIDE CONTROLS:")
        print("  Up/Down (Arrow keys): Forward/Backward")
        print("  A/D: Turn left/right")
        print("  Left/Right (Arrow keys): Roll left/right")
        print("  W/S: Up/Down")
        print("  T: Takeoff")
        print("  L: Land")
        print("  E: Emergency stop (stop all movement)")
        print("\nAI control will resume when no override keys are pressed")

        print("\nStarting in 3 seconds... Prepare for takeoff!")
        time.sleep(3)
        
        # Take off
        tello_controller.takeoff()
        
        while True:
            # Capture current view from Tello camera
            frame = tello_controller.capture_frame()
            
            if frame is None:
                print("Error: Failed to capture frame")
                time.sleep(1)
                continue
            
            # Display frame if debug is enabled
            if args.debug or args.show_frames:
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
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        
        if 'tello_controller' in locals():
            print("\nLanding drone and cleaning up...")
            tello_controller.stop()
            
    return 0

if __name__ == "__main__":
    sys.exit(main()) 