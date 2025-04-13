"""
Tello drone camera calibration tool.

This module provides a simple utility to calibrate the Tello drone camera
using a chessboard pattern. It captures multiple views of the chessboard
and calculates the camera matrix and distortion coefficients.
"""

import cv2
import numpy as np
import time
import os
import logging
from djitellopy import Tello

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class CameraCalibrationTool:
    """Tool for calibrating the Tello drone camera."""
    
    def __init__(self, chessboard_size=(6, 9), square_size=1.0):
        """
        Initialize the calibration tool.
        
        Args:
            chessboard_size: Size of the chessboard pattern (width, height)
            square_size: Size of each square in the chessboard (in arbitrary units)
        """
        self.logger = logging.getLogger('CameraCalibration')
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2) * square_size
        
        # Termination criteria for cornerSubPix
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None
        
        # Tello drone and frame
        self.drone = None
        self.frame = None
        self.required_samples = 15  # Number of good chessboard views required
        
        # Parameters for stream stability
        self.frame_warmup_count = 30  # Number of frames to discard for stream stabilization
        self.min_sample_interval = 1.0  # Minimum time between samples (seconds)
        self.stream_retry_count = 3  # Number of times to retry stream initialization
    
    def connect_to_drone(self):
        """Connect to the Tello drone."""
        retry_count = 0
        while retry_count < self.stream_retry_count:
            try:
                self.logger.info("Connecting to Tello drone...")
                self.drone = Tello()
                self.drone.connect()
                
                # Try to get battery level to verify connection
                battery = self.drone.get_battery()
                self.logger.info(f"Battery level: {battery}%")
                
                # Start video stream
                self.drone.streamon()
                self.logger.info("Video stream started, stabilizing...")
                
                # Wait for stream to initialize
                time.sleep(3)
                
                # Initialize frame reader
                self.frame_read = self.drone.get_frame_read()
                
                # Warm up the stream by discarding initial frames
                # This helps stabilize the H.264 decoder
                self._warmup_video_stream()
                
                self.logger.info("Connected to Tello drone successfully")
                return True
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Connection attempt {retry_count} failed: {e}")
                if self.drone:
                    try:
                        self.drone.streamoff()
                    except:
                        pass
                time.sleep(2)
        
        self.logger.error(f"Failed to connect after {self.stream_retry_count} attempts")
        return False
    
    def _warmup_video_stream(self):
        """Discard initial frames to stabilize the video stream."""
        if not self.frame_read:
            return
            
        self.logger.info(f"Warming up video stream ({self.frame_warmup_count} frames)...")
        frames_processed = 0
        start_time = time.time()
        
        # Create a progress bar window
        progress_frame = np.zeros((150, 400, 3), dtype=np.uint8)
        cv2.namedWindow("Stream Initialization", cv2.WINDOW_NORMAL)
        
        while frames_processed < self.frame_warmup_count:
            # Get a frame
            frame = self.frame_read.frame
            
            # Only count valid frames
            if frame is not None and frame.size > 0:
                frames_processed += 1
                
                # Update progress bar
                progress = int((frames_processed / self.frame_warmup_count) * 380)
                progress_frame.fill(0)
                cv2.rectangle(progress_frame, (10, 60), (10 + progress, 90), (0, 255, 0), -1)
                cv2.putText(progress_frame, f"Initializing stream: {frames_processed}/{self.frame_warmup_count}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Stream Initialization", progress_frame)
                cv2.waitKey(1)
            
            # Avoid infinite loop if frames aren't coming
            if time.time() - start_time > 15:  # 15 seconds timeout
                self.logger.warning("Warmup timeout reached, continuing anyway")
                break
                
            time.sleep(0.03)  # ~30fps
        
        cv2.destroyWindow("Stream Initialization")
        self.logger.info(f"Processed {frames_processed} warmup frames")
    
    def run_calibration(self):
        """Run the camera calibration process."""
        if not self.connect_to_drone():
            self.logger.error("Cannot proceed with calibration - drone not connected")
            return False
        
        self.logger.info(f"Starting calibration with {self.chessboard_size[1]}x{self.chessboard_size[0]} chessboard")
        self.logger.info("Press 'c' to capture when chessboard is detected")
        self.logger.info("Press 'q' to quit and calculate calibration")
        
        samples_collected = 0
        last_sample_time = time.time()
        valid_frame_count = 0
        total_frame_count = 0
        last_fps_update = time.time()
        fps = 0
        
        # Create calibration window with trackbar for corner detection parameters
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        
        # Parameters for adaptive detection
        chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        
        # Add additional flags to improve detection
        chessboard_flags += cv2.CALIB_CB_FAST_CHECK
        
        # Create a debug flag for visualization of the thresholded image
        debug_detection = False
        
        # Create trackbar for threshold adjustment
        cv2.createTrackbar("Threshold", "Calibration", 128, 255, lambda x: None)
        
        while samples_collected < self.required_samples:
            try:
                # Get frame from drone
                frame = self.frame_read.frame
                total_frame_count += 1
                
                # Skip empty or invalid frames
                if frame is None or frame.size == 0:
                    time.sleep(0.03)
                    continue
                
                # Convert color space (Tello returns RGB, OpenCV expects BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                valid_frame_count += 1
                
                # Create a copy of the frame for display
                display_frame = frame.copy()
                
                # Calculate FPS approximately every second
                current_time = time.time()
                if current_time - last_fps_update > 1.0:
                    fps = valid_frame_count / (current_time - last_fps_update)
                    valid_frame_count = 0
                    last_fps_update = current_time
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply preprocessing to improve contrast
                threshold_value = cv2.getTrackbarPos("Threshold", "Calibration")
                _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                
                # Try to find chessboard in the original grayscale image
                ret, corners = cv2.findChessboardCorners(
                    gray, 
                    (self.chessboard_size[1], self.chessboard_size[0]), 
                    chessboard_flags
                )
                
                # If not found, try with the thresholded image
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        thresh, 
                        (self.chessboard_size[1], self.chessboard_size[0]), 
                        chessboard_flags
                    )
                
                # If still not found, try with alternative pattern (ChArUco)
                # (This requires additional setup if you decide to use it)
                
                # Display current status
                status_text = f"Samples: {samples_collected}/{self.required_samples} | FPS: {fps:.1f} | Thresh: {threshold_value}"
                cv2.putText(display_frame, status_text, (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add debug visualization toggle
                cv2.putText(display_frame, "Press 'd' to toggle debug view", (20, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Debug view to help with troubleshooting
                if debug_detection:
                    # Display threshold image
                    debug_display = np.zeros((display_frame.shape[0], display_frame.shape[1]*2, 3), dtype=np.uint8)
                    debug_display[:, :display_frame.shape[1]] = display_frame
                    # Convert threshold image to BGR for display
                    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                    debug_display[:, display_frame.shape[1]:] = thresh_bgr
                    # Add label
                    cv2.putText(debug_display, "Binary Threshold", 
                               (display_frame.shape[1] + 20, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    display_frame = debug_display
                
                if ret:
                    # Draw the corners
                    cv2.drawChessboardCorners(
                        display_frame, 
                        (self.chessboard_size[1], self.chessboard_size[0]), 
                        corners, 
                        ret
                    )
                    
                    # Add instruction text
                    cv2.putText(display_frame, "Chessboard detected! Press 'c' to capture", 
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show confidence - how clear the detection is
                    coverage = 100 * (cv2.contourArea(corners) / (frame.shape[0] * frame.shape[1]))
                    cv2.putText(display_frame, f"Coverage: {coverage:.1f}%", 
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Capture on 'c' key press, ensuring minimum time between samples
                    if key == ord('c') and (current_time - last_sample_time) > self.min_sample_interval:
                        # Refine corner positions
                        corners2 = cv2.cornerSubPix(
                            gray, corners, (11, 11), (-1, -1), self.criteria
                        )
                        
                        # Store the object and image points
                        self.objpoints.append(self.objp)
                        self.imgpoints.append(corners2)
                        
                        samples_collected += 1
                        last_sample_time = current_time
                        
                        # Highlight that this frame was captured
                        cv2.putText(display_frame, "CAPTURED!", (frame.shape[1]//2 - 70, 
                                    frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1.2, (0, 0, 255), 3)
                        
                        self.logger.info(f"Captured sample {samples_collected}/{self.required_samples} (Coverage: {coverage:.1f}%)")
                        
                        # Display the captured frame a bit longer
                        cv2.imshow("Calibration", display_frame)
                        cv2.waitKey(500)  # Show "CAPTURED!" for 500ms
                else:
                    # Display instructions when no chessboard is detected
                    cv2.putText(display_frame, "No chessboard detected", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Add suggestions for improving detection
                    cv2.putText(display_frame, "Suggestions: Move closer, ensure good lighting", (20, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(display_frame, "Use trackbar to adjust threshold", (20, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Display the current frame
                cv2.imshow("Calibration", display_frame)
                
                # Check for quit command and other keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if samples_collected >= 5:  # Ensure we have enough samples
                        self.logger.info(f"Quitting with {samples_collected} samples")
                        break
                    else:
                        self.logger.warning("Need at least 5 samples for calibration")
                        cv2.putText(display_frame, "Need at least 5 samples!", (20, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Calibration", display_frame)
                        cv2.waitKey(1000)  # Show warning for 1 second
                
                # Toggle debug mode on 'd' press
                elif key == ord('d'):
                    debug_detection = not debug_detection
                    self.logger.info(f"Debug mode: {debug_detection}")
                    
                # Handle chessboard size adjustment
                elif key == ord('s'):
                    # Prompt for new chessboard size
                    self.logger.info("Current size: {}x{}".format(
                        self.chessboard_size[1], self.chessboard_size[0]))
                    self.logger.info("Adjust in code if needed (limitation of this UI)")
            
            except Exception as e:
                self.logger.warning(f"Frame processing error: {e}")
                time.sleep(0.1)
        
        # Calculate camera calibration if we have enough samples
        if samples_collected >= 5:
            self.logger.info("Calculating camera calibration...")
            
            # Display "Processing..." message
            processing_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(processing_frame, "Processing calibration...", 
                        (640//2 - 150, 480//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Calibration", processing_frame)
            cv2.waitKey(1)
            
            try:
                # Calculate camera calibration
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    self.objpoints, self.imgpoints, gray.shape[::-1], None, None
                )
                
                if ret:
                    self.camera_matrix = mtx
                    self.dist_coeffs = dist
                    self.calibration_error = ret
                    
                    # Calculate reprojection error
                    mean_error = 0
                    for i in range(len(self.objpoints)):
                        imgpoints2, _ = cv2.projectPoints(
                            self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
                        )
                        error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                        mean_error += error
                    
                    if len(self.objpoints) > 0:
                        mean_error /= len(self.objpoints)
                        self.logger.info(f"Calibration complete. Reprojection error: {mean_error}")
                        
                        # Show calibration results
                        result_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(result_frame, "Calibration Successful!", 
                                    (640//2 - 150, 480//2 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(result_frame, f"Reprojection error: {mean_error:.4f}", 
                                    (640//2 - 150, 480//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Show error interpretation
                        quality = "Excellent"
                        if mean_error > 1.0:
                            quality = "Poor"
                        elif mean_error > 0.5:
                            quality = "Average"
                        elif mean_error > 0.2:
                            quality = "Good"
                            
                        cv2.putText(result_frame, f"Calibration quality: {quality}", 
                                    (640//2 - 150, 480//2 + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.putText(result_frame, "Press any key to continue", 
                                    (640//2 - 150, 480//2 + 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow("Calibration", result_frame)
                        cv2.waitKey(0)
                        
                        return True
            except Exception as e:
                self.logger.error(f"Calibration computation error: {e}")
                result_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(result_frame, "Calibration Failed!", 
                            (640//2 - 150, 480//2 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_frame, str(e), 
                            (640//2 - 150, 480//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(result_frame, "Press any key to continue", 
                            (640//2 - 150, 480//2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Calibration", result_frame)
                cv2.waitKey(0)
                return False
        else:
            self.logger.warning("Not enough samples collected for calibration")
            return False
        
        return False
    
    def save_calibration(self, filename="tello_camera_calibration.xml"):
        """
        Save calibration parameters to a file.
        
        Args:
            filename: Name of the file to save calibration parameters
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.logger.error("No calibration data to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save calibration data
            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", self.camera_matrix)
            fs.write("dist_coeffs", self.dist_coeffs)
            fs.write("calibration_error", self.calibration_error)
            fs.write("chessboard_size", np.array(self.chessboard_size, dtype=np.int32))
            fs.write("square_size", self.square_size)
            fs.write("calibration_date", time.strftime("%Y-%m-%d %H:%M:%S"))
            fs.release()
            
            self.logger.info(f"Calibration saved to {filename}")
            
            # Also save human-readable version for reference
            np.savetxt(filename.replace('.xml', '_matrix.txt'), self.camera_matrix)
            np.savetxt(filename.replace('.xml', '_distortion.txt'), self.dist_coeffs)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
            return False
    
    def test_calibration(self, num_frames=100):
        """
        Test the calibration by showing undistorted frames.
        
        Args:
            num_frames: Number of frames to display in test
            
        Returns:
            bool: True if test was successful, False otherwise
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.logger.error("No calibration data to test")
            return False
        
        self.logger.info("Testing calibration with undistorted frames...")
        self.logger.info("Press 'q' to quit the test")
        
        frame_count = 0
        test_start_time = time.time()
        valid_frames = 0
        
        while frame_count < num_frames:
            try:
                # Get frame from drone
                frame = self.frame_read.frame
                if frame is None or frame.size == 0:
                    time.sleep(0.03)
                    continue
                
                # Convert color space (Tello returns RGB, OpenCV expects BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                valid_frames += 1
                
                # Undistort the frame
                h, w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
                )
                undistorted = cv2.undistort(
                    frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx
                )
                
                # Apply ROI if available
                x, y, w, h = roi
                if w > 0 and h > 0:
                    undistorted = undistorted[y:y+h, x:x+w]
                    # Resize to match original frame size for side-by-side display
                    undistorted = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))
                
                # Create side-by-side display
                h, w = frame.shape[:2]
                combined = np.zeros((h, w*2, 3), dtype=np.uint8)
                combined[:, :w] = frame
                combined[:, w:] = undistorted
                
                # Add labels
                cv2.putText(combined, "Original", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(combined, "Undistorted", (w + 20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(combined, f"Frame {frame_count+1}/{num_frames}", (20, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Calculate and display FPS
                elapsed = time.time() - test_start_time
                if elapsed > 0:
                    fps = valid_frames / elapsed
                    cv2.putText(combined, f"FPS: {fps:.1f}", (w + 20, h - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display the combined frame
                cv2.imshow("Calibration Test", combined)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_count += 1
            except Exception as e:
                self.logger.warning(f"Error during calibration test: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        return True
    
    def cleanup(self):
        """Clean up resources."""
        if self.drone:
            try:
                self.drone.streamoff()
                cv2.destroyAllWindows()
                self.logger.info("Cleanup completed")
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")


def main():
    """Main function to run the calibration tool."""
    # Create the calibration tool with 6x9 chessboard
    calibration_tool = CameraCalibrationTool(chessboard_size=(6, 9))
    
    try:
        # Run calibration
        calibration_success = calibration_tool.run_calibration()
        
        if calibration_success:
            # Save calibration
            calibration_file = "config/tello_camera_calibration.xml"
            save_success = calibration_tool.save_calibration(calibration_file)
            
            if save_success:
                # Test calibration
                calibration_tool.test_calibration()
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    finally:
        # Clean up
        calibration_tool.cleanup()


if __name__ == "__main__":
    main() 