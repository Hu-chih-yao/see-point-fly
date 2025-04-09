"""
Emergency Handler module for Tello drones.

This module provides emergency handling functionality for critical safety situations.
"""

import logging
import threading
import time

class EmergencyHandler:
    """
    Handler for emergency situations with the Tello drone.
    
    Centralizes emergency response actions and safety monitoring.
    """
    
    def __init__(self, drone_controller):
        """
        Initialize the emergency handler.
        
        Args:
            drone_controller: DroneController instance to control
        """
        self.logger = logging.getLogger('EmergencyHandler')
        self.drone_controller = drone_controller
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 2.0  # Check every 2 seconds
        
        # Safety thresholds
        self.min_battery_percent = 15  # Emergency land below this battery percentage
        self.critical_battery_percent = 10  # Emergency stop below this battery percentage
        self.min_height = 20  # Minimum safe height in cm
        self.max_height = 200  # Maximum safe height in cm
        
        # Actions history for response time calculation
        self.last_action_time = 0
        self.action_response_times = []
    
    def start_monitoring(self):
        """Start safety monitoring in a background thread."""
        if self.monitoring:
            self.logger.warning("Emergency monitoring already active")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Emergency monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3.0)
            self.logger.info("Emergency monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop that checks drone state for emergency conditions."""
        self.logger.debug("Emergency monitor loop started")
        
        while self.monitoring:
            try:
                if not self.drone_controller.is_connected:
                    time.sleep(self.monitor_interval)
                    continue
                
                # Check battery level
                self._check_battery()
                
                # Check if drone is flying
                if self.drone_controller.drone.is_flying:
                    # Check height
                    self._check_height()
                    
                    # Check flight time
                    self._check_flight_time()
                
                # Sleep before next check
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in emergency monitor loop: {e}")
                time.sleep(1.0)  # Wait on error
    
    def _check_battery(self):
        """Check battery level and trigger emergency actions if needed."""
        try:
            battery = self.drone_controller.get_battery()
            
            if battery <= self.critical_battery_percent:
                self.logger.critical(f"CRITICAL BATTERY LEVEL: {battery}%")
                self.emergency_stop("Critical battery level")
            elif battery <= self.min_battery_percent:
                self.logger.warning(f"LOW BATTERY LEVEL: {battery}%")
                self.emergency_land("Low battery level")
        except Exception as e:
            self.logger.error(f"Battery check failed: {e}")
    
    def _check_height(self):
        """Check if drone height is within safe limits."""
        try:
            height = self.drone_controller.get_height()
            
            if height > self.max_height:
                self.logger.warning(f"EXCEEDING MAXIMUM HEIGHT: {height}cm")
                self.emergency_land("Maximum height exceeded")
        except Exception as e:
            self.logger.error(f"Height check failed: {e}")
    
    def _check_flight_time(self):
        """Check if drone flight time is excessive."""
        try:
            flight_time = self.drone_controller.get_flight_time()
            
            # Log long flight times but don't take action
            if flight_time > 300:  # 5 minutes
                self.logger.info(f"Extended flight time: {flight_time}s")
        except Exception as e:
            self.logger.error(f"Flight time check failed: {e}")
    
    def emergency_stop(self, reason):
        """
        Immediate emergency stop - cuts all motors.
        
        Args:
            reason: Reason for emergency stop
        """
        self.logger.critical(f"EMERGENCY STOP: {reason}")
        
        try:
            start_time = time.time()
            self.drone_controller.drone.emergency()
            response_time = time.time() - start_time
            
            self.action_response_times.append(('emergency_stop', response_time))
            self.logger.info(f"Emergency stop executed in {response_time:.3f}s")
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
    
    def emergency_land(self, reason):
        """
        Emergency landing - attempts controlled landing.
        
        Args:
            reason: Reason for emergency landing
        """
        self.logger.warning(f"EMERGENCY LANDING: {reason}")
        
        try:
            start_time = time.time()
            
            # Stop all movement first
            self.drone_controller.drone.send_rc_control(0, 0, 0, 0)
            
            # Then initiate landing
            self.drone_controller.drone.land()
            
            response_time = time.time() - start_time
            self.action_response_times.append(('emergency_land', response_time))
            self.logger.info(f"Emergency landing initiated in {response_time:.3f}s")
        except Exception as e:
            self.logger.error(f"Emergency landing failed: {e}")
            
            # If controlled landing fails, try emergency stop as a last resort
            try:
                self.logger.critical("Trying emergency stop as fallback")
                self.drone_controller.drone.emergency()
            except Exception as ex:
                self.logger.error(f"Emergency fallback failed: {ex}")
    
    def get_response_stats(self):
        """
        Get statistics about emergency response times.
        
        Returns:
            dict: Statistics about response times
        """
        if not self.action_response_times:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}
            
        response_times = [t[1] for t in self.action_response_times]
        return {
            "count": len(response_times),
            "avg": sum(response_times) / len(response_times),
            "min": min(response_times),
            "max": max(response_times)
        } 