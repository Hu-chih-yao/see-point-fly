# Phase 1.2: Tello Drone Emergency Controls

This document details the implementation of emergency controls and manual override functionality for the Tello drone as part of Phase 1.2 of the Tello Drone Control System.

## Overview

Phase 1.2 focuses on implementing emergency controls and manual override capabilities for the Tello drone. This phase builds on the basic connectivity established in Phase 1.1 and adds critical safety features to ensure proper control and emergency response.

## Key Components

### 1. Keyboard Controller

The `KeyboardController` class provides comprehensive manual control over the drone through keyboard inputs, allowing for both normal operation and emergency override.

#### Core Functionality

- **Comprehensive Control**: Maps keyboard keys to all four primary flight controls:
  - **Roll**: Left/right horizontal movement (A/D keys)
  - **Pitch**: Forward/backward movement (W/S keys)
  - **Throttle**: Up/down vertical movement (Up/Down arrow keys)
  - **Yaw**: Left/right rotation (Left/Right arrow keys)

- **Flight Functions**:
  - Takeoff ('E' key)
  - Land ('X' key)
  - Emergency stop ('Q' key)
  - Reset RC controls ('R' key)

- **Speed Modes**:
  - Slow (30% speed) - '1' key
  - Normal (50% speed) - '2' key
  - Fast (80% speed) - '3' key

#### Implementation Details

The keyboard controller uses a threaded approach to maintain responsive control:

1. **Keyboard Listener Thread**: Monitors keyboard input and updates key states
2. **Control Loop Thread**: Continuously applies the current control values

```python
def _control_loop(self):
    """Main control loop that updates drone commands based on key states."""
    self.logger.debug("Control loop started")
    
    while self.is_active:
        try:
            # Get the current speed factor
            speed_factor = self.speed_factors[self.speed_mode]
            
            # Update velocities based on key states
            self._update_velocities(speed_factor)
            
            # Handle function keys (takeoff, land, etc.)
            self._handle_function_keys()
            
            # Send control commands to the drone if connected
            if self.drone_controller.is_connected:
                self.drone_controller.drone.send_rc_control(
                    self.left_right_velocity,
                    self.forward_backward_velocity,
                    self.up_down_velocity,
                    self.yaw_velocity
                )
            
            # Sleep to maintain control rate
            time.sleep(self.control_interval)
            
        except Exception as e:
            self.logger.error(f"Error in control loop: {e}")
            time.sleep(0.5)  # Wait longer on error
```

### 2. Emergency Handler

The `EmergencyHandler` class centralizes safety-critical functionality and provides automated monitoring for dangerous conditions.

#### Core Functionality

- **Safety Monitoring**: Continuously checks for dangerous conditions such as:
  - Critical battery levels
  - Excessive height
  - Extended flight time

- **Emergency Actions**:
  - **Emergency Stop**: Immediately cuts all motors (via `drone.emergency()`)
  - **Emergency Land**: Attempts controlled landing with fallback to emergency stop

- **Response Time Tracking**: Measures and logs the response time for emergency actions to validate system performance

#### Implementation Details

```python
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
```

## Integration with Main Application

The main application now integrates both the Keyboard Controller and Emergency Handler:

1. Initializes the drone connection
2. Starts the emergency monitoring system
3. Launches the keyboard controller for manual input
4. Provides a clean shutdown procedure for all components

```python
# Initialize emergency handler
logger.info("Initializing emergency handler...")
emergency_handler = EmergencyHandler(controller)
emergency_handler.start_monitoring()

# Initialize keyboard controller
logger.info("Initializing keyboard controller...")
keyboard_controller = KeyboardController(controller)
keyboard_controller.start()
```

## Key Design Considerations

### 1. Thread Safety and Resource Management

The system uses multiple threads to handle different aspects of drone control:

- **Keyboard Listening**: Non-blocking monitoring of keyboard inputs
- **Control Loop**: Regular updates to drone movement commands
- **Emergency Monitoring**: Background safety checks

All threads are properly managed with:
- **Daemon Mode**: All background threads are set as daemon threads to allow clean program exit
- **Thread Joining**: Proper cleanup of threads during shutdown
- **Thread-safe State Management**: Clear separation of state between threads

### 2. Emergency Response Strategy

The system implements a layered emergency response strategy:

1. **User-Triggered Emergency**: Direct action via 'Q' key for immediate motor cutoff
2. **Condition-Based Actions**: Automatic responses to dangerous conditions
3. **Fallback Mechanisms**: If primary emergency action fails, system attempts alternative actions

### 3. Control Responsiveness

To ensure responsive control:

- **High-Frequency Control Loop**: The control loop runs at 20Hz (50ms intervals) for smooth response
- **Velocity Scaling**: Different speed modes to accommodate precision vs. speed needs
- **Key State Tracking**: Continuous tracking of key states to handle multiple simultaneous inputs

## Testing and Validation

Testing for Phase 1.2 focused on:

1. **Response Time**: Measuring the delay between emergency command and drone response
2. **Reliability**: Validating consistent emergency behavior across different situations
3. **Control Feel**: Ensuring intuitive and responsive manual control
4. **Exception Handling**: Testing recovery from various error conditions

## Usage Example

To use the emergency control system:

1. Start the application with `python main.py`
2. When the keyboard controller activates, the following controls are available:

```
=== Keyboard Controls ===
Takeoff:           'e'
Land:              'x'
Emergency Stop:    'q'
Reset RC:          'r'
Forward/Backward:  'w'/'s'
Left/Right:        'a'/'d'
Up/Down:           'up'/'down' arrow keys
Yaw Left/Right:    'left'/'right' arrow keys
Speed (30%):       '1'
Speed (50%):       '2'
Speed (80%):       '3'
Exit Program:      'ctrl+c'
```

3. Press 'Q' at any time for immediate emergency stop

## Next Steps

With the emergency controls now implemented, the next phase (Phase 1.3) will focus on:

1. Enhanced telemetry display
2. More comprehensive state monitoring
3. Flight data logging and analysis

## Technical Limitations and Considerations

1. **Key Command Latency**: There is an inherent latency in the communication between key press and drone response, typically 100-200ms.

2. **Emergency Stop Implications**: The emergency stop command cuts all motor power immediately, which will cause the drone to fall from its current height. Use only in critical situations.

3. **Platform Considerations**: The keyboard control system is designed for desktop computers and may not be suitable for mobile or embedded deployments.

This implementation provides a robust emergency control system for the Tello drone, focusing on both manual override capability and automated safety monitoring. 