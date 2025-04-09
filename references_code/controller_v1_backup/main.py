"""
Main application entry point for the Tello Drone Control System.

This module demonstrates basic drone connectivity and manual control via keyboard.
"""

import time
import sys
import logging
import cv2
import numpy as np
import os
from control_system.drone_controller import DroneController
from control_system.keyboard_controller import KeyboardController
from safety.emergency_handler import EmergencyHandler

def main():
    """Main function to test drone connectivity and manual control."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('main')
    
    controller = None
    keyboard_controller = None
    emergency_handler = None
    
    try:
        # Initialize drone controller
        logger.info("Initializing drone controller...")
        controller = DroneController()
        
        # Connect to the drone
        logger.info("Connecting to Tello drone...")
        if not controller.connect():
            logger.error("Failed to connect to the drone")
            return 1
        
        # Get and display basic drone information
        logger.info("Retrieving drone information...")
        battery = controller.get_battery()
        sdk_version = controller.get_sdk_version()
        temp = controller.get_temperature()
        
        # Display drone information
        logger.info("=== Drone Information ===")
        logger.info(f"Battery: {battery}%")
        logger.info(f"SDK Version: {sdk_version}")
        logger.info(f"Temperature: {temp}°C")
        
        # Initialize emergency handler
        logger.info("Initializing emergency handler...")
        emergency_handler = EmergencyHandler(controller)
        emergency_handler.start_monitoring()
        
        # Initialize keyboard controller
        logger.info("Initializing keyboard controller...")
        keyboard_controller = KeyboardController(controller)
        keyboard_controller.start()
        
        # Start video stream
        logger.info("Starting video stream...")
        video_enabled = controller.start_video_stream(resolution="720p")
        if not video_enabled:
            logger.warning("Could not start video stream, continuing without video")
        
        # Check for calibration file
        calibration_file = "config/tello_camera_calibration.xml"
        calibration_available = os.path.exists(calibration_file)
        calibration_enabled = False
        
        if calibration_available:
            logger.info("Camera calibration file found")
            calibration_enabled = controller.load_camera_calibration(calibration_file)
            logger.info(f"Camera calibration {'enabled' if calibration_enabled else 'failed to load'}")
        else:
            logger.info("No camera calibration file found. Run camera_utils/camera_calibration.py to create one.")
        
        # Display keyboard controls
        print("\n=== Keyboard Controls ===")
        print("Takeoff:           't'")
        print("Land:              'l'")
        print("Emergency Stop:    'q'")
        print("Reset RC:          'r'")
        print("Stop Movement:     'e'")
        print("Forward/Backward:  'w'/'s'")
        print("Left/Right:        'a'/'d'")
        print("Throttle Up/Down:  '↑'/'↓' arrow keys")
        print("Rotate Left/Right: '←'/'→' arrow keys")
        print("Get Height:        'h'")
        print("Get Battery:       'b'")
        print("Toggle Video:      'v'")
        print("Toggle Calibration: 'c'")
        print("Exit Program:      'esc' or 'ctrl+c'\n")
        
        # Create windows for UI
        cv2.namedWindow("Tello Control", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Tello Control", 100, 100)
        
        if video_enabled:
            cv2.namedWindow("Tello Camera Feed", cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("Tello Camera Feed", 650, 100)
        
        # Display simple control pad
        control_pad = create_control_pad(calibration_available)
        cv2.imshow("Tello Control", control_pad)
        
        # Main loop variables
        last_battery_check = time.time()
        show_video = video_enabled
        
        # Main loop for keyboard handling and video display
        logger.info("Ready for keyboard control. Press 'esc' to exit...")
        
        while True:
            # Handle video display
            if video_enabled and show_video:
                # Get frame with timeout (non-blocking)
                frame = controller.get_frame(timeout=0.05)
                
                if frame is not None:
                    # Add calibration status to frame if available
                    if calibration_available:
                        status = "CALIBRATED" if calibration_enabled else "UNCALIBRATED"
                        color = (0, 255, 0) if calibration_enabled else (0, 0, 255)
                        cv2.putText(frame, status, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Display video frame
                    cv2.imshow("Tello Camera Feed", frame)
                else:
                    # Display placeholder when no frame is available
                    show_no_signal()
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on ESC key
            if key == 27:  # ESC key
                logger.info("Exit requested with ESC key")
                break
            
            # Toggle video display with 'v' key
            if key == ord('v') and video_enabled:
                show_video = not show_video
                logger.info(f"Video display {'enabled' if show_video else 'disabled'}")
                if not show_video:
                    cv2.destroyWindow("Tello Camera Feed")
                else:
                    cv2.namedWindow("Tello Camera Feed", cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow("Tello Camera Feed", 650, 100)
            
            # Toggle calibration with 'c' key
            if key == ord('c') and calibration_available:
                calibration_enabled = not calibration_enabled
                controller.enable_calibration(calibration_enabled)
                logger.info(f"Camera calibration {'enabled' if calibration_enabled else 'disabled'}")
                
            # Handle other keys with keyboard controller
            if key != 255:  # A key was pressed
                keyboard_controller.handle_key(key)
            
            # Display battery every 30 seconds
            current_time = time.time()
            if current_time - last_battery_check > 30:
                battery = controller.get_battery()
                logger.info(f"Battery level: {battery}%")
                last_battery_check = current_time
                
            # Short sleep to avoid maxing out CPU
            time.sleep(0.01)
        
        # Cleanup CV2 windows
        cv2.destroyAllWindows()
        return 0
    
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        # Ensure clean shutdown
        logger.info("Shutting down...")
        
        # Close CV2 windows if they exist
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Stop keyboard controller
        if keyboard_controller:
            try:
                logger.info("Stopping keyboard controller...")
                keyboard_controller.stop()
            except Exception as e:
                logger.error(f"Error stopping keyboard controller: {e}")
        
        # Stop emergency handler
        if emergency_handler:
            try:
                logger.info("Stopping emergency handler...")
                emergency_handler.stop_monitoring()
            except Exception as e:
                logger.error(f"Error stopping emergency handler: {e}")
        
        # Disconnect from drone
        if controller and controller.is_connected:
            try:
                logger.info("Disconnecting from drone...")
                controller.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from drone: {e}")
        
        logger.info("Shutdown complete")

def create_control_pad(calibration_available=False):
    """Create a simple control pad image to show the keyboard controls."""
    # Create a black image
    pad = 255 * np.ones((430, 500, 3), dtype=np.uint8)
    
    # Add text for controls
    cv2.putText(pad, "Tello Drone Control", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(pad, "W: Forward  S: Backward", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "A: Left  D: Right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "↑/↓: Up/Down", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "←/→: Rotate Left/Right", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "T: Takeoff  L: Land", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "Q: Emergency Stop", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "E: Stop Movement  R: Reset", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "H: Height  B: Battery", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(pad, "V: Toggle Video", (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    if calibration_available:
        cv2.putText(pad, "C: Toggle Calibration", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    else:
        cv2.putText(pad, "Camera not calibrated", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    cv2.putText(pad, "ESC: Exit Program", (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add a simple border
    cv2.rectangle(pad, (20, 20), (480, 400), (0, 0, 0), 2)
    
    return pad

def show_no_signal():
    """Display a no signal screen when video is not available."""
    try:
        # Create a static noise pattern
        h, w = 480, 640
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        
        # Add "No Signal" text
        cv2.putText(noise, "NO SIGNAL", (w//2-80, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Tello Camera Feed", noise)
    except Exception:
        pass  # Silently fail if window doesn't exist

if __name__ == "__main__":
    sys.exit(main()) 