"""
Keyboard Controller module for Tello drones.

This module provides keyboard control for manually overriding drone actions
and implementing emergency safety functions. Uses a simplified direct approach.
"""

import logging
import time

class KeyboardController:
    """
    Keyboard controller for manual override of Tello drone.
    
    Simplified implementation that directly maps key codes to drone commands.
    """
    
    def __init__(self, drone_controller):
        """
        Initialize the keyboard controller.
        
        Args:
            drone_controller: DroneController instance to control
        """
        self.logger = logging.getLogger('KeyboardController')
        self.drone_controller = drone_controller
        self.is_active = False
        
        # Speed settings
        self.fb_speed = 40  # Forward/backward speed
        self.lf_speed = 40  # Left/right speed
        self.ud_speed = 50  # Up/down speed
        self.yaw_degree = 30  # Yaw rotation speed
        
        # Last command time for rate limiting
        self.last_command_time = 0
        self.command_interval = 0.05  # 50ms interval for smooth control
    
    def start(self):
        """Start keyboard controller."""
        self.is_active = True
        self.logger.info("Keyboard controller started")
    
    def stop(self):
        """Stop keyboard controller."""
        self.is_active = False
        self.send_rc_control(0, 0, 0, 0)  # Stop all movement on exit
        self.logger.info("Keyboard controller stopped")
    
    def handle_key(self, key_code):
        """
        Handle a key press with direct command mapping.
        
        Args:
            key_code: The key code from cv2.waitKey() or similar
        
        Returns:
            bool: True if the key was handled, False otherwise
        """
        if not self.is_active or not self.drone_controller.is_connected:
            return False
            
        # Rate limiting for smoother control
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return False
        
        self.last_command_time = current_time
        
        try:
            # Get character from key code
            key_char = chr(key_code).lower() if 0 < key_code < 255 else None
            
            # Movement commands
            if key_char == 'w':  # Forward
                self.send_rc_control(0, self.fb_speed, 0, 0)
                self.logger.debug("Forward")
                return True
                
            elif key_char == 's':  # Backward
                self.send_rc_control(0, -self.fb_speed, 0, 0)
                self.logger.debug("Backward")
                return True
                
            elif key_char == 'a':  # Left
                self.send_rc_control(-self.lf_speed, 0, 0, 0)
                self.logger.debug("Left")
                return True
                
            elif key_char == 'd':  # Right
                self.send_rc_control(self.lf_speed, 0, 0, 0)
                self.logger.debug("Right")
                return True
            
            # Special keys for arrow key handling
            elif key_code == 82 or key_code == 0:  # Up arrow
                self.send_rc_control(0, 0, self.ud_speed, 0)
                self.logger.debug("Throttle up")
                return True
                
            elif key_code == 84 or key_code == 1:  # Down arrow
                self.send_rc_control(0, 0, -self.ud_speed, 0)
                self.logger.debug("Throttle down")
                return True
                
            elif key_code == 81 or key_code == 2:  # Left arrow
                self.send_rc_control(0, 0, 0, -self.yaw_degree)
                self.logger.debug("Rotate left")
                return True
                
            elif key_code == 83 or key_code == 3:  # Right arrow
                self.send_rc_control(0, 0, 0, self.yaw_degree)
                self.logger.debug("Rotate right")
                return True
            
            # Function keys
            elif key_char == 't':  # Takeoff
                if not self.drone_controller.drone.is_flying:
                    self.logger.info("Taking off...")
                    self.drone_controller.drone.takeoff()
                return True
                
            elif key_char == 'l':  # Land
                if self.drone_controller.drone.is_flying:
                    self.logger.info("Landing...")
                    self.drone_controller.drone.land()
                return True
                
            elif key_char == 'q':  # Emergency stop
                self.logger.warning("EMERGENCY STOP TRIGGERED")
                self.drone_controller.drone.emergency()
                return True
                
            elif key_char == 'e':  # Stop all movement
                self.send_rc_control(0, 0, 0, 0)
                self.logger.debug("Stop all movement")
                return True
                
            elif key_char == 'r':  # Reset velocities
                self.send_rc_control(0, 0, 0, 0)
                self.logger.debug("Reset velocities")
                return True
                
            elif key_char == 'h':  # Get height
                height = self.drone_controller.drone.get_height()
                self.logger.info(f"Current height: {height}cm")
                return True
                
            elif key_char == 'b':  # Get battery
                battery = self.drone_controller.drone.get_battery()
                self.logger.info(f"Battery level: {battery}%")
                return True
                
            # Video control keys are handled in main.py, not here
            # This prevents duplicate handling
                
        except Exception as e:
            self.logger.error(f"Error handling key {key_code}: {e}")
        
        return False
    
    def send_rc_control(self, left_right, forward_backward, up_down, yaw):
        """Send RC control to the drone with error handling."""
        try:
            self.drone_controller.drone.send_rc_control(
                left_right, forward_backward, up_down, yaw
            )
        except Exception as e:
            self.logger.error(f"Error sending RC command: {e}") 