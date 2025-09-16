import cv2
import numpy as np
from .drone_space import DroneActionSpace, ActionPoint
from typing import List, Tuple, Optional
from ..clients.vlm_client import VLMClient
import os
import time
import yaml

class ActionProjector:
    """
    Base class for handling projection between 2D screen coordinates and 3D world space
    Maintains camera model and provides methods for point projection
    """

    def __init__(self,
                 image_width=960,
                 image_height=720,
                 config_path="config_tello.yaml"):
        """
        Initialize the projector with image dimensions and optional camera parameters

        Args:
            image_width (int): Width of the input image
            image_height (int): Height of the input image
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
        self.action_space = DroneActionSpace(n_samples=8)

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.api_provider = config.get('api_provider', 'gemini')
        self.custom_model = config.get('model_name', '').strip()

        # Determine model name based on provider
        model_name = self._determine_model_name()

        # Initialize VLM client
        self.vlm_client = VLMClient(self.api_provider, model_name)
        self.model_name = model_name

        print(f"[ActionProjector] Initialized with {self.api_provider} provider using model: {self.model_name}")

        # Initialize timestamp and output directory
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"action_visualizations/{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

    def _determine_model_name(self):
        """Determine model name based on provider and custom setting"""
        if self.custom_model:
            return self.custom_model

        # Default models based on provider
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

            # Convert to screen coordinates
            screen_x = int(center_x + x_projected)
            screen_y = int(center_y - z_projected)  # Negative because screen Y increases downward

            return (screen_x, screen_y)

        except Exception as e:
            print(f"Error in project_point: {e}")
            return (int(self.image_width/2), int(self.image_height/2))

    def reverse_project_point(self, point_2d: Tuple[int, int], depth: float = 1.0) -> Tuple[float, float, float]:
        """Project 2D screen point to 3D world space at given depth"""
        try:
            screen_x, screen_y = point_2d

            # Center points
            center_x = self.image_width / 2
            center_y = self.image_height / 2

            # Convert to normalized coordinates
            x_offset = screen_x - center_x
            y_offset = center_y - screen_y  # Flip Y axis

            # Calculate perspective scaling
            fov_factor = np.tan(np.radians(self.fov_horizontal / 2))

            # Scale by depth and FOV
            x = (x_offset / (self.image_width / 2)) * depth * fov_factor
            z = (y_offset / (self.image_height / 2)) * depth * fov_factor
            y = depth

            return (x, y, z)

        except Exception as e:
            print(f"Error in reverse_project_point: {e}")
            return (0.0, 1.0, 0.0)

    def get_vlm_points(self, image: np.ndarray, instruction: str, **kwargs) -> List[ActionPoint]:
        """Get points from VLM - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_vlm_points")

    def _get_single_action(self, image: np.ndarray, instruction: str, **kwargs) -> ActionPoint:
        """Get single next best action - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_single_action")

    def visualize_coordinate_system(self, image: Optional[np.ndarray] = None) -> np.ndarray:
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
