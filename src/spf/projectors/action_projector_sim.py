import cv2
import numpy as np
from ..spaces.drone_space_sim import DroneActionSpaceSim, ActionPoint
from typing import List, Tuple
from ..clients.vlm_client import VLMClient
import os
import time
import json
import yaml

class ActionProjectorSim:
    """
    Handles projection between 2D screen coordinates and 3D world space
    Maintains camera model and provides methods for point projection
    """

    def __init__(self,
                 image_width=1920,
                 image_height=1080,
                 adaptive_mode=False,
                 config_path="config_sim.yaml"):
        """
        Initialize the projector with image dimensions and optional camera parameters

        Args:
            image_width (int): Width of the input image (default: match monitor 1)
            image_height (int): Height of the input image (default: match monitor 1)
            adaptive_mode (bool): Enable adaptive depth-based movement scaling
            config_path (str): Path to configuration file
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fov_horizontal = 108  # degrees
        self.fov_vertical = 108    # degrees

        # Define coordinate space limits with wider range
        self.x_range = (-3.0, 3.0)    # Left/Right: wider range
        self.y_range = (0.5, 2.0)     # Forward depth: keep same for good perspective
        self.z_range = (-1.8, 1.8)    # Up/Down: 3x the original (-0.6, 0.6)

        # Calculate focal length
        self.focal_length = self.image_width / (2 * np.tan(np.radians(self.fov_horizontal/2)))

        # Initialize action space
        self.action_space = DroneActionSpaceSim(n_samples=8)

        # Store adaptive mode setting
        self.adaptive_mode = adaptive_mode

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.api_provider = config.get('api_provider', 'gemini')
        self.custom_model = config.get('model_name', '').strip()

        # Determine model name based on provider and custom setting
        model_name = self._determine_model_name()

        # Initialize VLM client
        self.vlm_client = VLMClient(self.api_provider, model_name)
        self.model_name = model_name

        print(f"[ActionProjectorSim] Initialized with {self.api_provider} provider using model: {self.model_name}")

        # Initialize timestamp and output directory
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"action_visualizations/{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

    def _determine_model_name(self):
        """Determine model name based on provider and custom setting"""
        if self.custom_model:
            return self.custom_model

        # Default models based on provider (simulator doesn't have mode variations)
        if self.api_provider == "openai":
            return "google/gemini-2.5-flash"
        else:  # gemini provider
            return "gemini-2.5-flash"

    def project_point(self, point_3d: Tuple[float, float, float]) -> Tuple[int, int]:
        """Project 3D point using proper perspective projection for drone view"""
        try:
            x, y, z = point_3d

            # Center points
            center_x = self.image_width / 2
            center_y = self.image_height / 2

            # Calculate perspective scaling based on field of view
            fov_factor = np.tan(np.radians(self.fov_horizontal / 2))

            # Perspective projection with proper FOV
            # y is our depth (forward distance)
            if y < 0.1:  # Avoid division by zero
                y = 0.1

            # Scale x and z based on perspective and FOV
            x_projected = (x / (y * fov_factor)) * (self.image_width / 2)
            z_projected = (z / (y * fov_factor)) * (self.image_height / 2)

            # Calculate final screen coordinates
            screen_x = int(center_x + x_projected)
            screen_y = int(center_y - z_projected)  # Negative because screen y increases downward

            # Clamp to image boundaries
            screen_x = max(0, min(screen_x, self.image_width-1))
            screen_y = max(0, min(screen_y, self.image_height-1))

            # Debug print - uncomment if needed
            # print(f"3D Point ({x:.2f}, {y:.2f}, {z:.2f}) -> Screen ({screen_x}, {screen_y})")

            return (screen_x, screen_y)
        except Exception as e:
            print(f"Error in project_point: {e}")
            # Return center of screen as fallback
            return (self.image_width // 2, self.image_height // 2)

    def reverse_project_point(self, point_2d: Tuple[int, int], depth: float = 1) -> Tuple[float, float, float]:
        """Project 2D image point back to 3D space"""
        # Center and normalize coordinates
        x_normalized = (point_2d[0] - self.image_width/2) / (self.image_width/2)
        y_normalized = (self.image_height/2 - point_2d[1]) / (self.image_height/2)

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
        - Depth 1: Set to 0 (no movement)
        - Depth 2-4: Set to 4 (moderate movement)
        - Depth 5+: Non-linear scaling for efficiency

        Args:
            vlm_depth: Depth value from VLM (1-10 scale)

        Returns:
            adjusted_depth: Custom scaled depth value
        """
        if vlm_depth <= 3:
            # Very close objects - no movement
            adjusted_depth = 0
            print(f"Simulator: VLM depth {vlm_depth}/10 → Adjusted depth {adjusted_depth} (No movement - too close)")
        else:  # vlm_depth >= 3
            # Far objects - non-linear scaling for efficiency
            base = (vlm_depth / 10.0)**2.4 * 5
            adjusted_depth = base
            print(f"Simulator: VLM depth {vlm_depth}/10 → Adjusted depth {adjusted_depth:.2f} (Non-linear scaling)")

        return adjusted_depth

    def get_vlm_points(self, image: np.ndarray, instruction: str) -> List[ActionPoint]:
        """Use VLM to identify points based on current mode and API provider"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # Get single action from VLM
            actions = [self._get_single_action(image, instruction)]

            if actions:
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

                    # Draw obstacles if present
                    if action.detected_obstacles:
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
                #cv2.imwrite(save_path, viz_image)

                # Save decision data
                decision_data = {
                    "timestamp": timestamp,
                    "instruction": instruction,
                    "actions": []
                }

                # Add action and obstacle data
                for action in actions:
                    action_data: dict = {
                        "dx": action.dx,
                        "dy": action.dy,
                        "dz": action.dz,
                        "screen_x": action.screen_x,
                        "screen_y": action.screen_y
                    }

                    # Add obstacles if present
                    if action.detected_obstacles:
                        action_data["obstacles"] = action.detected_obstacles

                    decision_data["actions"].append(action_data)

                with open(f"{self.output_dir}/decision_{timestamp}.json", 'w') as f:
                    json.dump(decision_data, f, indent=2)

            return actions

        except Exception as e:
            print(f"Error getting points: {e}")
            return []



    def _get_single_action(self, image: np.ndarray, instruction: str) -> ActionPoint:
        """Get single next best action"""

        # Create mode-specific prompts
        if self.adaptive_mode:
            prompt = f"""You are a drone navigation expert analyzing a simulator camera view.

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
- Your accuracy in point placement and depth estimation is critical for adaptive navigation"""
        else:
            prompt = f"""You are a drone navigation expert analyzing a simulator camera view.

Task: {instruction}

main task:
1. Identify objects in the image that match the description "{instruction}".
2. Then, select the MOST RELEVANT target object and place a "target point" DIRECTLY ON that object.
sub task:
3. Identify obstacles in the path, if necessary, "slighty" adjust the point.

Return in this exact JSON format:
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
- Aim for target's center."""

        '''
        Requirements:
            - Place the point PRECISELY where the drone should move next
            - Consider immediate obstacles and choose a safe path
            - If the target is a vehicle or structure, aim for its center
            - Identify ALL obstacles that could block the path with accurate bounding boxes
            - Your accuracy in point placement and obstacle detection is critical for safe navigation
        '''
        try:
            # Get response from API
            response_text = self.vlm_client.generate_response(prompt, image)

            # Parse response text - handle potential markdown formatting
            response_text = VLMClient.clean_response_text(response_text)

            print(f"\n{self.api_provider.upper()} Response:")
            print(response_text)

            # Mode-specific JSON parsing
            if self.adaptive_mode:
                # Adaptive mode - parse depth-based response
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

                # Use the adaptive depth adjustment
                adjusted_depth = self.calculate_adjusted_depth(vlm_depth)

                # Project 2D point to 3D with adaptive depth
                x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y), depth=adjusted_depth)

                # Create ActionPoint with adaptive depth info
                action = ActionPoint(
                    dx=x3d, dy=y3d, dz=z3d,
                    action_type="move",
                    screen_x=pixel_x,
                    screen_y=pixel_y
                )

                # Store depth info for action space
                action.adaptive_depth = adjusted_depth
                action.vlm_depth = vlm_depth

                print(f"\n[ADAPTIVE] Identified single action: {point_info['label']}")
                print(f"2D Normalized: ({x}, {y})")
                print(f"2D Pixels: ({pixel_x}, {pixel_y})")
                print(f"Depth estimation: {vlm_depth}/10 (adjusted to {adjusted_depth:.2f})")
                print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")

                return action
            else:
                # Obstacle mode - parse obstacle-based response
                response_data = json.loads(response_text)
                if not response_data:
                    raise ValueError("No data returned from VLM")

                # Get waypoint coordinates
                y, x = response_data['point']
                pixel_x = int((x / 1000.0) * self.image_width)
                pixel_y = int((y / 1000.0) * self.image_height)

                # Project 2D point to 3D with default depth
                x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y))

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

                print(f"\nIdentified single action: {response_data.get('label', 'move')}")
                print(f"2D Normalized: ({x}, {y})")
                print(f"2D Pixels: ({pixel_x}, {pixel_y})")
                print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")
                if action.detected_obstacles:
                    print(f"Detected {len(action.detected_obstacles)} obstacles")

                return action

        except Exception as e:
            print(f"Error in single action mode: {e}")
            print("Full response:")
            if 'response_text' in locals():
                print(response_text)
            else:
                print("No response received")
            # Return a default action instead of None
            return ActionPoint(
                dx=0.0, dy=0.0, dz=0.0,
                action_type="move",
                screen_x=self.image_width // 2,
                screen_y=self.image_height // 2
            )

    def visualize_coordinate_system(self, image: np.ndarray | None = None) -> np.ndarray:
        """Create a visualization of the coordinate system for debugging"""
        if image is None:
            # Create blank image
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        height, width = image.shape[:2]
        center = (width//2, height//2)

        # Draw coordinate axes
        cv2.line(image, center, (width, height//2), (0, 0, 255), 2)  # X axis (red)
        cv2.line(image, center, (width//2, 0), (0, 255, 0), 2)       # Y axis (green)
        cv2.line(image, center, (width//4, height//2), (255, 0, 0), 2) # Z axis (blue)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "X (right)", (width-100, height//2-10), font, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "Y (forward)", (width//2+10, 30), font, 0.6, (0, 255, 0), 2)
        cv2.putText(image, "Z (up)", (width//4-30, height//2-10), font, 0.6, (255, 0, 0), 2)

        # Add grid lines
        grid_spacing = 100
        alpha = 0.3  # Grid line opacity

        for i in range(0, width, grid_spacing):
            cv2.line(image, (i, 0), (i, height), (100, 100, 100), 1)
            if i % (grid_spacing*5) == 0:  # Darker lines every 500 pixels
                cv2.line(image, (i, 0), (i, height), (150, 150, 150), 2)
                cv2.putText(image, f"{i-center[0]}", (i, height-10), font, 0.4, (200, 200, 200), 1)

        for i in range(0, height, grid_spacing):
            cv2.line(image, (0, i), (width, i), (100, 100, 100), 1)
            if i % (grid_spacing*5) == 0:  # Darker lines every 500 pixels
                cv2.line(image, (0, i), (width, i), (150, 150, 150), 2)
                cv2.putText(image, f"{center[1]-i}", (10, i), font, 0.4, (200, 200, 200), 1)

        # Add sample points in 3D space
        sample_points = [
            (1.0, 1.0, 0.0),   # Right and forward
            (-1.0, 1.0, 0.0),  # Left and forward
            (0.0, 1.0, 1.0),   # Forward and up
            (0.0, 1.0, -1.0),  # Forward and down
            (0.0, 2.0, 0.0)    # Further forward
        ]

        for i, point in enumerate(sample_points):
            try:
                screen_point = self.project_point(point)
                cv2.circle(image, screen_point, 5, (0, 255, 255), -1)
                cv2.putText(image, f"P{i+1}: {point}", (screen_point[0]+5, screen_point[1]-5),
                           font, 0.4, (0, 255, 255), 1)
            except:
                pass

        # Add resolution and FOV info
        cv2.putText(image, f"Resolution: {width}x{height}, FOV: {self.fov_horizontal}°",
                   (10, height-10), font, 0.5, (255, 255, 255), 1)

        return image
