import cv2
import numpy as np
from ..base.action_projector import ActionProjector
from ..base.drone_space import ActionPoint
from .drone_space import TelloDroneActionSpace
from typing import List, Tuple
import os
import time
import json

class TelloActionProjector(ActionProjector):
    """
    Tello-specific action projector with mode-specific processing
    """

    def __init__(self,
                 image_width=960,
                 image_height=720,
                 mode="adaptive_mode",
                 config_path="config_tello.yaml"):
        """
        Initialize the Tello projector with mode-specific settings

        Args:
            image_width (int): Width of the input image
            image_height (int): Height of the input image
            mode (str): Operational mode ("adaptive_mode" or "obstacle_mode")
            config_path (str): Path to configuration file
        """
        # Store operational mode FIRST (needed by parent's _determine_model_name)
        self.operational_mode = mode
        
        super().__init__(image_width, image_height, config_path)

        # Use Tello-specific action space
        self.action_space = TelloDroneActionSpace(n_samples=8)

        print(f"[TelloActionProjector] Initialized in {mode} with {self.api_provider} provider using model: {self.model_name}")

    def _determine_model_name(self):
        """Determine model name based on provider, mode, and custom setting"""
        if self.custom_model:
            return self.custom_model

        # Default models based on provider and mode
        if self.api_provider == "openai":
            if self.operational_mode == "obstacle_mode":
                return "google/gemini-2.5-pro"
            else:
                return "google/gemini-2.5-flash"
        else:  # gemini provider
            if self.operational_mode == "obstacle_mode":
                return "gemini-2.5-pro"
            else:
                return "gemini-2.0-flash"

    def reverse_project_point(self, point_2d: Tuple[int, int], depth: float = 2) -> Tuple[float, float, float]:
        """Project 2D image point back to 3D space with Tello-specific parameters"""
        # Set reference point at 35% from top of frame
        reference_y = self.image_height * 0.35

        # Center and normalize coordinates
        x_normalized = (point_2d[0] - self.image_width/2) / (self.image_width/2)
        y_normalized = (reference_y - point_2d[1]) / (self.image_height/2)

        # Adjust depth based on vertical position (closer if lower in image)
        depth_factor = 1.0 + (y_normalized * 0.5)  # Adjust depth based on height
        depth = depth * depth_factor

        # Calculate 3D coordinates with optimized depth
        x = depth * x_normalized * np.tan(np.radians(self.fov_horizontal/2))
        z = depth * y_normalized * np.tan(np.radians(self.fov_vertical/2))
        y = depth

        return (x, y, z)

    def calculate_adjusted_depth(self, vlm_depth):
        """
        Custom depth adjustment with specific rules:
        - Depth <= 2: Use minimal depth for calculations but flag for yaw-only
        - Depth > 2: Non-linear scaling for efficiency

        Args:
            vlm_depth: Depth value from VLM (1-10 scale)

        Returns:
            tuple: (adjusted_depth, yaw_only_flag)
        """
        if vlm_depth <= 2:
            # Very close objects - use minimal depth for calculations, but flag as yaw-only
            adjusted_depth = 0.5  # Minimal depth to avoid calculation issues
            yaw_only = True
            print(f"Tello: VLM depth {vlm_depth}/10 → Adjusted depth {adjusted_depth} (YAW ONLY - too close)")
            return adjusted_depth, yaw_only
        else:  # vlm_depth > 2
            # Far objects - non-linear scaling for efficiency
            base = (vlm_depth / 10.0)**2.0 * 8
            adjusted_depth = base
            yaw_only = False
            print(f"Tello: VLM depth {vlm_depth}/10 → Adjusted depth {adjusted_depth:.2f} (Normal movement)")
            return adjusted_depth, yaw_only

    def get_vlm_points(self, image: np.ndarray, instruction: str, tello_controller=None) -> List[ActionPoint]:
        """Use VLM to identify points based on current mode and API provider"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # Get single action from VLM with mode-specific processing
            if self.operational_mode == "obstacle_mode":
                print("\nin obstacle mode")
                actions = [self._get_single_action(image, instruction, tello_controller)]
            else:
                actions = [self._get_single_action(image, instruction)]

            if actions:
                print("\n actions in visualization part:")
                print("/n", actions)

                # Save visualization
                viz_image = image.copy()

                # Draw points on image
                for i, action in enumerate(actions, 1):
                    # Draw point
                    cv2.circle(viz_image,
                              (int(action.screen_x), int(action.screen_y)),
                              10, (0, 255, 0), -1)

                    # Add label
                    cv2.putText(
                        viz_image,
                        f"{i}: ({action.dx:.1f}, {action.dy:.1f}, {action.dz:.1f})",
                        (int(action.screen_x) + 15, int(action.screen_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    # Draw obstacles if present (obstacle_mode only)
                    if (self.operational_mode == "obstacle_mode" and
                        hasattr(action, 'detected_obstacles') and action.detected_obstacles):
                        for obstacle in action.detected_obstacles:
                            if 'bounding_box' in obstacle:
                                ymin, xmin, ymax, xmax = obstacle['bounding_box']
                                # Draw rectangle for obstacle
                                cv2.rectangle(viz_image,
                                            (int(xmin), int(ymin)),
                                            (int(xmax), int(ymax)),
                                            (0, 0, 255), 2)  # Red color for obstacles
                                # Add obstacle label
                                label = obstacle.get('label', 'obstacle')
                                cv2.putText(viz_image, label,
                                        (int(xmin), int(ymin)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)

                # Save visualization
                save_path = f"{self.output_dir}/decision_{timestamp}.jpg"
                cv2.imwrite(save_path, viz_image)

                # Save decision data
                decision_data = {
                    "timestamp": timestamp,
                    "mode": self.operational_mode,
                    "instruction": instruction,
                    "actions": []
                }

                # Add action and obstacle data
                for action in actions:
                    action_data = {
                            "dx": action.dx,
                            "dy": action.dy,
                            "dz": action.dz,
                            "screen_x": action.screen_x,
                            "screen_y": action.screen_y
                        }

                    # Add obstacles if present (obstacle_mode only)
                    if (self.operational_mode == "obstacle_mode" and
                        hasattr(action, 'detected_obstacles') and action.detected_obstacles):
                        action_data["obstacles"] = action.detected_obstacles

                    decision_data["actions"].append(action_data)

                with open(f"{self.output_dir}/decision_{timestamp}.json", 'w') as f:
                    json.dump(decision_data, f, indent=2)

            return actions

        except Exception as e:
            print(f"Error getting points: {e}")
            return []

    def _get_single_action(self, image: np.ndarray, instruction: str, tello_controller=None) -> ActionPoint:
        """Get single next best action with mode-specific processing"""

        # Mode-specific processing
        if self.operational_mode == "obstacle_mode":
            # Enhanced obstacle-aware processing
            print("\nFinished encoding image")
            print(f"[{self.api_provider.upper()}] Preparing API call at {time.strftime('%H:%M:%S')}")
            api_start_time = time.time()

            # Ensure intensive keepalive is active right before the API call
            if tello_controller:
                print(f"[{self.api_provider.upper()}] Confirming intensive keepalive before API call")
                tello_controller.start_intensive_keepalive()

            prompt = f"""You are a drone navigation expert analyzing a drone camera view.

        Task: {instruction}

        main task:
        1. Identify objects in the image that match the description "{instruction}".
        2. Then, select the MOST RELEVANT target object and place a "target point" DIRECTLY ON that object.
        sub task:
        3. Identify obstacles in the path, if necessary, "slighty" adjust the point.

        Return in this JSON format:
        {{
            "point": [y, x],
            "label": "action description",
            "obstacles": [
                    {{"bounding_box": [ymin, xmin, ymax, xmax], "label": "obstacle_description"}}
            ]
        }}

        Coordinate system:
        - x: 0-1000 scale (500=center, >500=right, <500=left)
        - y: 0-1000 scale (lower values=higher in image/sky)

        Notes:
        - "Pointing on the target" is the most important thing.
        - Prioritize the closest/largest matching object if multiple exist
        - Consider immediate obstacles and choose a safe path.
        - Aim for target's center.
        """
        else:
            # Adaptive mode - original behavior
            prompt = f"""You are a drone navigation expert analyzing a drone camera view.

            Task: {instruction}

            First, identify ALL objects in the image that match the description "{instruction}".
            Then, select the MOST RELEVANT target object and place a single point DIRECTLY ON that object.

            Return in this exact JSON format:
            [{{"point": [y, x], "depth": depth_value, "label": "action description"}}]

            Coordinate system:
            - x: 0-1000 scale (500=center, >500=right, <500=left)
            - y: 0-1000 scale (lower values=higher in image/sky)
            - depth: 1-10 scale where:
                * 1: Object is very close/large in frame
                * 10: Object is far away/small in frame

            IMPORTANT:
            - Place the point PRECISELY on the center of the target object
            - Choose the largest/closest matching object if multiple exist
            - Assess the depth based on how much of the frame the object occupies
            - Your accuracy in point placement is critical for navigation success"""

        try:
            # Get response from API
            if self.operational_mode == "obstacle_mode":
                print(f"[{self.api_provider.upper()}] Sending API request at {time.strftime('%H:%M:%S')}")

            response_text = self.vlm_client.generate_response(prompt, image)

            if self.operational_mode == "obstacle_mode":
                api_duration = time.time() - api_start_time
                print(f"[{self.api_provider.upper()}] Response received in {api_duration:.2f} seconds")

                # API call complete, can go back to normal keepalive if needed
                if tello_controller:
                    tello_controller.stop_intensive_keepalive()

            # Parse response text - handle potential markdown formatting
            from ..clients.vlm_client import VLMClient
            response_text = VLMClient.clean_response_text(response_text)

            print(f"\n{self.api_provider.upper()} Response:")
            print(response_text)

            # Mode-specific JSON parsing
            if self.operational_mode == "obstacle_mode":
                try:
                    # Parse JSON response for obstacle mode
                    response_data = json.loads(response_text)
                    if not response_data:
                        raise ValueError("No data returned from VLM")

                    # Convert normalized coordinates to pixel coordinates
                    y, x = response_data['point']
                    pixel_x = int((x / 1000.0) * self.image_width)
                    pixel_y = int((y / 1000.0) * self.image_height)

                    # Project 2D point to 3D (obstacle mode uses default depth)
                    x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y), depth=1.1)

                    # Create ActionPoint
                    action = ActionPoint(
                        dx=x3d, dy=y3d, dz=z3d,
                        action_type="move",
                        screen_x=pixel_x,
                        screen_y=pixel_y
                    )

                    # Add obstacles if present
                    if 'obstacles' in response_data:
                        obstacles = []
                        for obstacle in response_data['obstacles']:
                            if 'bounding_box' in obstacle:
                                ymin, xmin, ymax, xmax = obstacle['bounding_box']
                                # Convert to pixel coordinates if normalized
                                if max(obstacle['bounding_box']) <= 1000:
                                    xmin = int((xmin / 1000.0) * self.image_width)
                                    ymin = int((ymin / 1000.0) * self.image_height)
                                    xmax = int((xmax / 1000.0) * self.image_width)
                                    ymax = int((ymax / 1000.0) * self.image_height)
                                obstacle['bounding_box'] = [ymin, xmin, ymax, xmax]
                            obstacles.append(obstacle)
                        action.detected_obstacles = obstacles

                    print(f"\nIdentified single action: {response_data.get('label')}")
                    print(f"2D Normalized: ({x}, {y})")
                    print(f"2D Pixels: ({pixel_x}, {pixel_y})")
                    print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")
                    if hasattr(action, 'detected_obstacles') and action.detected_obstacles:
                        print(f"Detected {len(action.detected_obstacles)} obstacles")

                    return action

                except json.JSONDecodeError as json_error:
                    print(f"[{self.api_provider.upper()}] Error parsing JSON: {json_error}")
                    print(f"[{self.api_provider.upper()}] Raw response text: {response_text}")

                    # Try to manually extract the point information using regex
                    import re
                    point_match = re.search(r'"point":\s*\[(\d+),\s*(\d+)\]', response_text)
                    if point_match:
                        print(f"[{self.api_provider.upper()}] Attempting fallback point extraction with regex")
                        y, x = int(point_match.group(1)), int(point_match.group(2))
                        pixel_x = int((x / 1000.0) * self.image_width)
                        pixel_y = int((y / 1000.0) * self.image_height)
                        x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y), depth=1.1)

                        # Create basic ActionPoint without obstacles
                        action = ActionPoint(
                            dx=x3d, dy=y3d, dz=z3d,
                            action_type="move",
                            screen_x=pixel_x,
                            screen_y=pixel_y
                        )
                        print(f"[{self.api_provider.upper()}] Fallback action created: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")
                        return action

                    raise
            else:
                # Adaptive mode - original JSON parsing with depth
                points_data = json.loads(response_text)
                if not points_data:
                    raise ValueError("No points returned from VLM")

                # Take first (and should be only) point
                point_info = points_data[0]

                # Convert normalized coordinates to pixel coordinates
                y, x = point_info['point']
                pixel_x = int((x / 1000.0) * self.image_width)
                pixel_y = int((y / 1000.0) * self.image_height)

                # Get depth from VLM's response (default to 4 if not provided)
                vlm_depth = point_info.get('depth', 4)

                # Use the new depth adjustment that returns both depth and yaw-only flag
                adjusted_depth, yaw_only = self.calculate_adjusted_depth(vlm_depth)

                # Project 2D point to 3D with custom depth
                x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y), depth=adjusted_depth)

                # Create ActionPoint with yaw_only flag
                action = ActionPoint(
                    dx=x3d, dy=y3d, dz=z3d,
                    action_type="move",
                    screen_x=pixel_x,
                    screen_y=pixel_y,
                    yaw_only=yaw_only  # Set the yaw-only flag based on depth
                )

                print(f"\nIdentified single action: {point_info['label']}")
                print(f"2D Normalized: ({x}, {y})")
                print(f"2D Pixels: ({pixel_x}, {pixel_y})")
                print(f"Depth estimation: {vlm_depth}/10 (adjusted to {adjusted_depth:.2f})")
                if yaw_only:
                    print(f"[SAFETY] YAW ONLY mode - object too close for forward movement")
                print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")

                return action

        except Exception as e:
            if self.operational_mode == "obstacle_mode":
                print(f"[{self.api_provider.upper()}] Error in API call: {e}")
                if 'response_text' in locals():
                    print(f"[{self.api_provider.upper()}] Full response:")
                    print(response_text)
                else:
                    print(f"[{self.api_provider.upper()}] No response received from API")
            else:
                print(f"Error in single action mode: {e}")
                print("Full response:")
                if 'response_text' in locals():
                    print(response_text)
            return None
