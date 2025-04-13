"""
Drone Controller module for Tello drones.

This module provides a DroneController class that handles basic drone connectivity
and commands for the Tello drone.
"""

from djitellopy import Tello
import threading
import time
import logging
import cv2
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DroneController:
    """
    Main controller class for the Tello drone.
    
    Handles connection, basic commands, video streaming and keepalive functionality.
    """
    
    def __init__(self):
        """Initialize the drone controller with default values."""
        self.logger = logging.getLogger('DroneController')
        self.drone = Tello()
        self.is_connected = False
        self.keepalive_thread = None
        self.keepalive_interval = 10  # seconds
        self.last_command_time = 0
        
        # Video stream attributes
        self.video_thread = None
        self.is_streaming = False
        self.current_frame = None
        self.frame_ready = threading.Event()
        self.frame_read = None
        
        # Camera calibration attributes
        self.camera_matrix = None
        self.dist_coeffs = None
        self.use_calibration = False
    
    def connect(self):
        """
        Establish connection with the Tello drone.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info("Connecting to Tello drone...")
            self.drone.connect()
            self.is_connected = True
            self.last_command_time = time.time()
            
            # Verify connection with a simple command
            battery = self.drone.get_battery()
            self.logger.info(f"Initial battery level: {battery}%")
            
            # Start the keepalive thread if connection successful
            self.start_keepalive()
            self.logger.info("Successfully connected to Tello drone")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Tello drone: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Safely disconnect from the drone and clean up resources."""
        self.logger.info("Disconnecting from Tello drone...")
        self.is_connected = False
        
        # Stop video stream if active
        if self.is_streaming:
            self.stop_video_stream()
        
        # Wait for keepalive thread to terminate
        if self.keepalive_thread and self.keepalive_thread.is_alive():
            self.keepalive_thread.join(timeout=2.0)
            
        # Make sure the drone is landed
        if self.drone.is_flying:
            try:
                self.drone.land()
                time.sleep(1)  # Give the drone time to process the land command
            except Exception as e:
                self.logger.error(f"Error landing drone during disconnect: {e}")
        
        self.logger.info("Drone disconnected")
    
    def load_camera_calibration(self, filename="config/tello_camera_calibration.xml"):
        """
        Load camera calibration parameters from a file.
        
        Args:
            filename: Path to the XML file containing camera calibration parameters
            
        Returns:
            bool: True if calibration loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filename):
                self.logger.warning(f"Calibration file not found: {filename}")
                return False
                
            self.logger.info(f"Loading camera calibration from {filename}")
            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
            
            self.camera_matrix = fs.getNode("camera_matrix").mat()
            self.dist_coeffs = fs.getNode("dist_coeffs").mat()
            
            fs.release()
            
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                self.use_calibration = True
                self.logger.info("Camera calibration loaded successfully")
                return True
            else:
                self.use_calibration = False
                self.logger.warning("Failed to load valid calibration parameters")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading camera calibration: {e}")
            self.use_calibration = False
            return False
    
    def enable_calibration(self, enable=True):
        """
        Enable or disable the use of camera calibration.
        
        Args:
            enable: Whether to enable calibration
            
        Returns:
            bool: True if calibration is now enabled, False otherwise
        """
        if enable and (self.camera_matrix is None or self.dist_coeffs is None):
            self.logger.warning("Cannot enable calibration: No calibration data loaded")
            return False
            
        self.use_calibration = enable
        self.logger.info(f"Camera calibration {'enabled' if enable else 'disabled'}")
        return self.use_calibration
    
    def start_keepalive(self):
        """Start a background thread to send keepalive signals."""
        if self.is_connected:
            self.keepalive_thread = threading.Thread(
                target=self._keepalive_worker,
                daemon=True
            )
            self.keepalive_thread.start()
            self.logger.debug("Keepalive thread started")
    
    def _keepalive_worker(self):
        """Send periodic keepalive commands to prevent auto-landing when flying."""
        self.logger.debug("Keepalive worker started")
        while self.is_connected:
            try:
                current_time = time.time()
                time_since_last_command = current_time - self.last_command_time
                
                # Only send keepalive if:
                # 1. The drone is actually flying, or
                # 2. We haven't sent any command in a while (prevent disconnection)
                if self.drone.is_flying or time_since_last_command > self.keepalive_interval:
                    # Use a non-keepalive command for grounded drones to maintain connection
                    if self.drone.is_flying:
                        self.drone.send_keepalive()
                        self.logger.debug("Keepalive signal sent (flying)")
                    else:
                        # For non-flying drones, use a command that won't cause issues
                        # get_battery is reliable and won't affect drone state
                        self.drone.get_battery()
                        self.logger.debug("Connection maintenance command sent (not flying)")
                    
                    self.last_command_time = current_time
            except Exception as e:
                self.logger.warning(f"Keepalive communication issue: {e}")
                # If keepalive fails, wait a shorter time before retrying
                time.sleep(2)
                continue
                
            # Sleep for the specified interval
            time.sleep(self.keepalive_interval)
    
    def start_video_stream(self, resolution="720p"):
        """
        Start video streaming with specified resolution.
        
        Args:
            resolution: Resolution setting ("720p", "480p", etc.)
            
        Returns:
            bool: True if video stream started successfully, False otherwise
        """
        if not self.is_connected:
            self.logger.warning("Cannot start video stream: drone not connected")
            return False
        
        if self.is_streaming:
            self.logger.info("Video stream already active")
            return True
            
        try:
            # Set resolution before starting stream
            if resolution == "720p":
                self.drone.set_video_resolution(self.drone.RESOLUTION_720P)
            elif resolution == "480p":
                self.drone.set_video_resolution(self.drone.RESOLUTION_480P)
            else:
                self.logger.warning(f"Unknown resolution: {resolution}, using default")
            
            # Try to load camera calibration
            calibration_file = "config/tello_camera_calibration.xml"
            if os.path.exists(calibration_file):
                self.load_camera_calibration(calibration_file)
            
            # Start the Tello video stream
            self.drone.streamon()
            self.logger.info(f"Video stream started at {resolution}")
            
            # Create the frame reader
            self.frame_read = self.drone.get_frame_read()
            
            # Start video processing thread
            self.is_streaming = True
            self.video_thread = threading.Thread(
                target=self._video_worker,
                daemon=True
            )
            self.video_thread.start()
            self.logger.debug("Video worker thread started")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to start video stream: {e}")
            return False
    
    def _video_worker(self):
        """Background thread for video processing."""
        self.logger.debug("Video worker started")
        frame_count = 0
        last_log_time = time.time()
        
        while self.is_connected and self.is_streaming:
            try:
                # Check if frame reader is working
                if self.frame_read is None or self.frame_read.stopped:
                    self.logger.warning("Frame reader not working, attempting to restart...")
                    self.frame_read = self.drone.get_frame_read()
                    time.sleep(0.5)
                    continue
                
                # Get the latest frame (this clears the buffer automatically)
                frame = self.frame_read.frame
                
                # Process frame if valid
                if frame is not None:
                    # Convert color space for OpenCV processing (RGB to BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Apply calibration to undistort if enabled
                    if self.use_calibration and self.camera_matrix is not None and self.dist_coeffs is not None:
                        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
                    
                    # Store the frame and signal that it's ready
                    self.current_frame = frame
                    self.frame_ready.set()
                    
                    # Count frames for FPS calculation
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - last_log_time
                    
                    # Log FPS approximately once per second
                    if elapsed > 1.0:
                        fps = frame_count / elapsed
                        self.logger.debug(f"Video FPS: {fps:.2f}")
                        frame_count = 0
                        last_log_time = current_time
                
                # Limit frame processing rate to reduce CPU usage
                # ~30 FPS is smooth enough for most applications
                time.sleep(0.03)
                
            except Exception as e:
                self.logger.error(f"Error in video worker: {e}")
                time.sleep(0.1)  # Short delay on error before retrying
    
    def get_frame(self, timeout=1.0, apply_calibration=None):
        """
        Get the latest frame from the video stream.
        
        Args:
            timeout: Maximum time to wait for a frame in seconds
            apply_calibration: Override global calibration setting for this frame
                (None = use global setting, True = force calibration, False = no calibration)
            
        Returns:
            numpy.ndarray: Video frame as a NumPy array, or None if no frame available
        """
        if not self.is_connected or not self.is_streaming or self.current_frame is None:
            return None
        
        # Wait for a new frame to be available
        if self.frame_ready.wait(timeout=timeout):
            self.frame_ready.clear()  # Reset the event for next frame
            frame = self.current_frame.copy()  # Get a copy of the current frame
            
            # Apply calibration if requested specifically for this frame and not already applied
            if apply_calibration is True and not self.use_calibration and \
               self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
            
            return frame
        
        # Timeout occurred, return the current frame anyway
        if self.current_frame is not None:
            frame = self.current_frame.copy()
            
            # Apply calibration if requested specifically for this frame
            if apply_calibration is True and not self.use_calibration and \
               self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
                
            return frame
        
        return None
    
    def stop_video_stream(self):
        """Stop the video stream."""
        if not self.is_connected or not self.is_streaming:
            return
            
        try:
            # Signal video thread to stop
            self.is_streaming = False
            
            # Stop Tello video stream
            self.drone.streamoff()
            
            # Clean up resources
            if self.frame_read is not None:
                self.frame_read = None
            
            # Video thread will terminate automatically (daemon)
            self.logger.info("Video stream stopped")
        except Exception as e:
            self.logger.error(f"Error stopping video stream: {e}")
    
    def execute_command(self, command_func, *args, **kwargs):
        """
        Execute a command and update the last command timestamp.
        
        Args:
            command_func: The drone command function to execute
            *args, **kwargs: Arguments to pass to the command function
            
        Returns:
            The result of the command function
        """
        result = command_func(*args, **kwargs)
        self.last_command_time = time.time()
        return result
    
    def get_battery(self):
        """
        Get the current battery percentage of the drone.
        
        Returns:
            int: Battery percentage (0-100)
        """
        if not self.is_connected:
            self.logger.warning("Cannot get battery: drone not connected")
            return -1
            
        try:
            battery = self.drone.get_battery()
            self.last_command_time = time.time()  # Update last command time
            self.logger.info(f"Battery level: {battery}%")
            return battery
        except Exception as e:
            self.logger.error(f"Failed to get battery level: {e}")
            return -1
    
    def get_temperature(self):
        """
        Get the current temperature of the drone.
        
        Returns:
            int: Temperature in celsius
        """
        if not self.is_connected:
            self.logger.warning("Cannot get temperature: drone not connected")
            return -1
            
        try:
            # The Tello API doesn't have separate high/low temperature methods
            # Just return the single temperature value
            temp = self.drone.get_temperature()
            self.last_command_time = time.time()  # Update last command time
            self.logger.info(f"Temperature: {temp}°C")
            return temp
        except Exception as e:
            self.logger.error(f"Failed to get temperature: {e}")
            return -1
    
    def get_height(self):
        """
        Get the current height of the drone in cm.
        
        Returns:
            int: Height in centimeters
        """
        if not self.is_connected:
            self.logger.warning("Cannot get height: drone not connected")
            return -1
            
        try:
            height = self.drone.get_height()
            self.last_command_time = time.time()  # Update last command time
            self.logger.info(f"Current height: {height}cm")
            return height
        except Exception as e:
            self.logger.error(f"Failed to get height: {e}")
            return -1
    
    def get_flight_time(self):
        """
        Get the flight time in seconds.
        
        Returns:
            int: Flight time in seconds
        """
        if not self.is_connected:
            self.logger.warning("Cannot get flight time: drone not connected")
            return -1
            
        try:
            flight_time = self.drone.get_flight_time()
            self.last_command_time = time.time()  # Update last command time
            self.logger.info(f"Flight time: {flight_time}s")
            return flight_time
        except Exception as e:
            self.logger.error(f"Failed to get flight time: {e}")
            return -1

    def get_sdk_version(self):
        """
        Get the SDK version of the drone.
        
        Returns:
            str: SDK version string
        """
        if not self.is_connected:
            self.logger.warning("Cannot get SDK version: drone not connected")
            return "Unknown"
            
        try:
            # The correct method for getting SDK version
            version = self.drone.query_sdk_version()
            self.last_command_time = time.time()  # Update last command time
            self.logger.info(f"SDK Version: {version}")
            return version
        except Exception as e:
            self.logger.error(f"Failed to get SDK version: {e}")
            return "Unknown" 