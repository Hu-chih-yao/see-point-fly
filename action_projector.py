import cv2
import numpy as np
from drone_space import DroneActionSpace, ActionPoint
from typing import Dict, List, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import os
import base64
import time
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ActionProjector:
    """
    Handles projection between 2D screen coordinates and 3D world space
    Maintains camera model and provides methods for point projection
    """
    
    def __init__(self, 
                 image_width=960,
                 image_height=720,
                 camera_matrix=None,
                 dist_coeffs=None):
        """
        Initialize the projector with image dimensions and optional camera parameters
        
        Args:
            image_width (int): Width of the input image (default: match monitor 1)
            image_height (int): Height of the input image (default: match monitor 1)
            camera_matrix (np.ndarray): Optional 3x3 camera matrix
            dist_coeffs (np.ndarray): Optional distortion coefficients
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
        
        # Initialize Gemini
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        # Initialize timestamp and output directory
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"action_visualizations/{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Add mode flag
        self.mode = "waypoint"  # or "single"

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
    '''unknown usage
    def sample_action_point(self) -> ActionPoint:
        """Sample action points that make sense for drone navigation"""
        while True:
            # Sample points in a more natural range for drone movement
            x = np.random.uniform(-1.5, 1.5)     # Narrower left/right range
            y = np.random.uniform(1.0, 2.5)      # Forward distance (not too close/far)
            z = np.random.uniform(-0.5, 0.5)     # Smaller height range
            
            point_2d = self.project_point((x, y, z))
            
            # Validate the projected point is in a reasonable screen position
            if (0 <= point_2d[0] < self.image_width and 
                self.image_height * 0.2 <= point_2d[1] <= self.image_height * 0.8):
                return ActionPoint(x, y, z, "move")

    def generate_action_visualization(self, image: np.ndarray) -> Tuple[np.ndarray, List[ActionPoint]]:
        """Generate visualization with proper hemisphere sampling"""
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Save original input image
        cv2.imwrite(f"{self.output_dir}/input_frame.jpg", image)
        
        # Draw coordinate system
        self.draw_coordinate_system(annotated_image)
        center_point = (width//2, height//2)
        
        # Draw debug grid (optional)
        grid_spacing = 100
        for i in range(0, width, grid_spacing):
            cv2.line(annotated_image, (i, 0), (i, height), (100, 100, 100), 1)
        for i in range(0, height, grid_spacing):
            cv2.line(annotated_image, (0, i), (width, i), (100, 100, 100), 1)
        
        # Sample and draw actions
        valid_actions = []
        attempts = 0
        max_attempts = 50
        
        while len(valid_actions) < 8 and attempts < max_attempts:
            action = self.sample_action_point()
            point_2d = self.project_point((action.dx, action.dy, action.dz))
            action.screen_x = point_2d[0]
            action.screen_y = point_2d[1]
            
            if (width*0.1 <= point_2d[0] <= width*0.9 and 
                height*0.1 <= point_2d[1] <= height*0.8):
                
                # Save individual action visualization
                action_viz = image.copy()
                self.draw_action_visualization(
                    action_viz, action, len(valid_actions) + 1,
                    start_point=center_point)
                cv2.imwrite(
                    f"{self.output_dir}/action_{len(valid_actions)+1}.jpg", 
                    cv2.cvtColor(action_viz, cv2.COLOR_RGB2BGR)
                )
                
                # Add to main visualization
                self.draw_action_visualization(
                    annotated_image, action, len(valid_actions) + 1,
                    start_point=center_point)
                
                # Draw debug info
                cv2.putText(annotated_image, 
                           f"({action.dx:.1f}, {action.dy:.1f}, {action.dz:.1f})", 
                           (int(action.screen_x) + 30, int(action.screen_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                valid_actions.append(action)
                
                print(f"\nAction {len(valid_actions)}:")
                print(f"  3D: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})")
                print(f"  Screen: ({point_2d[0]}, {point_2d[1]})")
                print(f"  Height from center: {height//2 - point_2d[1]}")
            
            attempts += 1
        
        # Save final visualization
        cv2.imwrite(f"{self.output_dir}/final_visualization.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Save action data as text
        self._save_action_data(valid_actions)
        
        return annotated_image, valid_actions

    def draw_coordinate_system(self, image: np.ndarray) -> np.ndarray:
        """Draw coordinate system axes"""
        height, width = image.shape[:2]
        center = (width//2, height//2)  # Center of image
        
        # Draw axes through center
        cv2.line(image, (0, height//2), (width, height//2), (0, 0, 255), 1)  # X axis
        cv2.line(image, (width//2, 0), (width//2, height), (0, 0, 255), 1)  # Z axis
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "X", (width-50, height//2-20), font, 0.8, (0, 0, 255), 2)
        cv2.putText(image, "Z", (width//2+20, 30), font, 0.8, (0, 0, 255), 2)
        
        return image
    
    def _save_action_data(self, actions: List[ActionPoint]) -> None:
        """Save action data in a readable format"""
        with open(f"{self.output_dir}/action_data.txt", "w") as f:
            f.write(f"Action Visualization Data - {self.timestamp}\n")
            f.write("-" * 50 + "\n\n")
            
            for i, action in enumerate(actions, 1):
                f.write(f"Action {i}:\n")
                f.write(f"  3D: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})\n")
                f.write(f"  Screen: ({action.screen_x}, {action.screen_y})\n")
                f.write("\n")
    '''

    ''' arrow method ( not being used for now)
    def draw_arrow(self, image: np.ndarray, start_pt: Tuple[int, int], 
                  end_pt: Tuple[int, int], color: Tuple[int, int, int], 
                  thickness: int) -> np.ndarray:
        """Draw an arrow with proper head size"""
        # Calculate arrow length
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        arrow_length = np.sqrt(dx*dx + dy*dy)
        
        if arrow_length < 1:
            return image
        
        # Calculate arrow head size (proportional to length)
        arrow_head_length = min(30, arrow_length * 0.3)
        arrow_head_angle = np.pi / 6  # 30 degrees
        
        # Draw main line
        cv2.line(image, start_pt, end_pt, color, thickness)
        
        # Draw arrow head
        if arrow_length >= 20:  # Only draw head if arrow is long enough
            angle = np.arctan2(dy, dx)
            for sign in [-1, 1]:
                head_angle = angle + sign * arrow_head_angle
                head_x = int(end_pt[0] - arrow_head_length * np.cos(head_angle))
                head_y = int(end_pt[1] - arrow_head_length * np.sin(head_angle))
                cv2.line(image, end_pt, (head_x, head_y), color, thickness)
        
        return image

    def draw_action_visualization(self, image: np.ndarray, action: ActionPoint, 
                                index: int, start_point: Tuple[int, int]) -> np.ndarray:
        """Draw a single action visualization"""
        color = (0, 255, 0)  # Green
        end_point = (int(action.screen_x), int(action.screen_y))
        
        # Draw arrow from start point to action point
        self.draw_arrow(image, start_point, end_point, color, 2)
        
        # Draw marker circle
        cv2.circle(image, end_point, 20, (255, 255, 255), -1)  # White background
        cv2.circle(image, end_point, 20, color, 2)  # Green border
        
        # Add number label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(index)
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
        text_x = end_point[0] - text_size[0]//2
        text_y = end_point[1] + text_size[1]//2
        
        # Draw text with outline for better visibility
        cv2.putText(image, text, (text_x, text_y), font, 0.8, (0, 0, 0), 3)  # Black outline
        cv2.putText(image, text, (text_x, text_y), font, 0.8, color, 2)  # Green text
        
        return image

    def get_llm_choice(self, image: np.ndarray, actions: List[ActionPoint], instruction: str) -> Tuple[ActionPoint, str]:
        """Get LLM to choose an action based on the visualization"""
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Construct prompt
        prompt = f"""You are a drone navigation expert. I'm showing you a drone's view with numbered green arrows.
                    Each arrow represents a possible movement direction for the drone.

                    Task: {instruction}

                    The arrows are numbered 1-{len(actions)}. Each arrow shows:
                    - Direction: Where the drone will move
                    - Length: How far it will move
                    - Position: The final position after movement

                    Choose ONE arrow that best accomplishes the task.
                    Consider:
                    1. Which direction gets closest to the goal
                    2. Choose efficient movements

                    Format your response exactly as:
                    CHOSEN_ARROW: <number>
                    REASON: <brief explanation>
                    """

        # Send to Gemini
        response = self.model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': encoded_image}
        ])
        
        # Parse response
        chosen_number = None
        reason = ""
        for line in response.text.split('\n'):
            if line.startswith('CHOSEN_ARROW:'):
                try:
                    chosen_number = int(line.split(':')[1].strip())
                except:
                    continue
            elif line.startswith('REASON:'):
                reason = line.split(':')[1].strip()
        
        if chosen_number is None or chosen_number < 1 or chosen_number > len(actions):
            raise ValueError("Invalid action choice from LLM")
            
        return actions[chosen_number-1], reason
    '''
    
    def reverse_project_point(self, point_2d: Tuple[int, int], depth: float = 2) -> Tuple[float, float, float]:
        """Project 2D image point back to 3D space"""
        # Set reference point at 20% from top of frame
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

    def set_mode(self, mode: str):
        """Set operation mode: 'waypoint' or 'single'"""
        if mode not in ["waypoint", "single"]:
            raise ValueError("Mode must be 'waypoint' or 'single'")
        self.mode = mode

    def get_gemini_points(self, image: np.ndarray, instruction: str) -> List[ActionPoint]:
        """Use Gemini to identify points based on current mode"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # Get points from Gemini
            if self.mode == "waypoint":
                actions = self._get_waypoint_path(image, instruction)
            else:
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
                
                # Save visualization
                save_path = f"{self.output_dir}/decision_{timestamp}.jpg"
                cv2.imwrite(save_path, viz_image)
                
                # Save decision data
                decision_data = {
                    "timestamp": timestamp,
                    "mode": self.mode,
                    "instruction": instruction,
                    "actions": [
                        {
                            "dx": action.dx,
                            "dy": action.dy,
                            "dz": action.dz,
                            "screen_x": action.screen_x,
                            "screen_y": action.screen_y
                        }
                        for action in actions
                    ]
                }
                
                with open(f"{self.output_dir}/decision_{timestamp}.json", 'w') as f:
                    json.dump(decision_data, f, indent=2)
            
            return actions
            
        except Exception as e:
            print(f"Error getting points: {e}")
            return []

    def _get_waypoint_path(self, image: np.ndarray, instruction: str) -> List[ActionPoint]:
        """Get sequence of waypoints for path planning"""
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"""You are a drone navigation expert. Looking at this drone camera view:

            Task: {instruction}

            Return 3 waypoints in JSON format to guide the drone:
            [{{"point": [y, x], "label": "waypoint description"}}]

            Important Guidelines:
            - x: 500 represents center of view, higher values = right, lower values = left
            - y: Lower values = higher in image (closer to sky)
            - For flying to a target:
            1. First point: Slightly up and forward (y: 300-400, x: 500)
            2. Second point: Align with target horizontally
            3. Final point: At the target location

            Example for "fly to building straight ahead":
            [
                {{"point": [350, 500], "label": "Take off and move forward"}},
                {{"point": [400, 500], "label": "Continue straight ahead"}},
                {{"point": [450, 500], "label": "Arrive at building"}}
            ]
            """

        try:
            # Get response from Gemini
            response = self.model.generate_content([
                prompt,
                {'mime_type': 'image/jpeg', 'data': encoded_image}
            ])

            # Parse response text - handle potential markdown formatting
            response_text = response.text
            if "```json" in response_text:
                # Extract JSON from markdown code block
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                # Extract from generic code block
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            print("\nGemini Response:")
            print(response_text)
            
            # Parse JSON response
            points_data = json.loads(response_text)
            actions = []
            
            for point_info in points_data:
                # Convert normalized coordinates to pixel coordinates
                y, x = point_info['point']
                pixel_x = int((x / 1000.0) * self.image_width)
                pixel_y = int((y / 1000.0) * self.image_height)
                
                # Project 2D point to 3D
                x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y))
                
                # Create ActionPoint
                action = ActionPoint(
                    dx=x3d, dy=y3d, dz=z3d,
                    action_type="move",
                    screen_x=pixel_x,
                    screen_y=pixel_y
                )
                actions.append(action)
                
                print(f"\nIdentified point: {point_info['label']}")
                print(f"2D Normalized: ({x}, {y})")
                print(f"2D Pixels: ({pixel_x}, {pixel_y})")
                print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")
            
            return actions
            
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            print("Full response:")
            print(response.text)
            return []

    def calculate_adjusted_depth(self, gemini_depth):
        """
        Non-linear depth adjustment that makes:
        - Close objects (1-3) move slower for precision
        - Far objects (7-10) move faster for efficiency
        
        Args:
            gemini_depth: Depth value from Gemini (1-10 scale)
            
        Returns:
            adjusted_depth: Non-linear scaled depth value
        """
        # Base scaling with curve
        base = (gemini_depth / 10.0)**1.8 * 6.0
        
        # Add minimum threshold to prevent too slow movements
        adjusted_depth = max(0.5, base)
        
        print(f"Gemini depth {gemini_depth}/10 → Adjusted depth {adjusted_depth:.2f}")
        return adjusted_depth

    def _get_single_action(self, image: np.ndarray, instruction: str) -> ActionPoint:
        """Get single next best action"""
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        

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
            # Get response from Gemini
            response = self.model.generate_content([
                prompt,
                {'mime_type': 'image/jpeg', 'data': encoded_image}
            ])

            # Parse response text - handle potential markdown formatting
            response_text = response.text
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            print("\nGemini Response:")
            print(response_text)
            
            # Parse JSON response
            points_data = json.loads(response_text)
            if not points_data:
                raise ValueError("No points returned from Gemini")
            
            # Take first (and should be only) point
            point_info = points_data[0]
            
            # Convert normalized coordinates to pixel coordinates
            y, x = point_info['point']
            pixel_x = int((x / 1000.0) * self.image_width)
            pixel_y = int((y / 1000.0) * self.image_height)
            
            # Get depth from Gemini's response (default to 4 if not provided)
            gemini_depth = point_info.get('depth', 4)  # Default to middle range if not specified
            
            # Use the new non-linear depth adjustment
            adjusted_depth = self.calculate_adjusted_depth(gemini_depth)
            
            # Project 2D point to 3D with custom depth
            x3d, y3d, z3d = self.reverse_project_point((pixel_x, pixel_y), depth=adjusted_depth)
            

            # Create ActionPoint
            action = ActionPoint(
                dx=x3d, dy=y3d, dz=z3d,
                action_type="move",
                screen_x=pixel_x,
                screen_y=pixel_y
            )
            
            print(f"\nIdentified single action: {point_info['label']}")
            print(f"2D Normalized: ({x}, {y})")
            print(f"2D Pixels: ({pixel_x}, {pixel_y})")
            print(f"Depth estimation: {gemini_depth}/10 (adjusted to {adjusted_depth:.2f})")
            print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")
            
            return action
            
        except Exception as e:
            print(f"Error in single action mode: {e}")
            print("Full response:")
            print(response.text)
            return None

    ''' points testing code
    def test_spatial_understanding(self, image: np.ndarray, instruction: str, mode: str = "waypoint"):
        """Test the spatial understanding system in specified mode"""
        self.set_mode(mode)
        print(f"\n=== Testing Spatial Understanding System ({self.mode.upper()} mode) ===")
        print(f"Instruction: {instruction}")
        
        # Get points from Gemini
        actions = self.get_gemini_points(image, instruction)
        
        if not actions:
            print("No points identified by Gemini")
            return
        
        # Create visualizations based on mode
        if self.mode == "waypoint":
            self._visualize_waypoint_path(image, actions)
        else:
            if actions[0] is None:  # Add safety check
                print("Error: No valid single action returned")
                return
            self._visualize_single_action(image, actions[0])

    def _visualize_waypoint_path(self, image: np.ndarray, actions: List[ActionPoint]):
        """Visualize complete path with waypoints"""
        # ... existing waypoint visualization code ...

    def _visualize_single_action(self, image: np.ndarray, action: ActionPoint):
        """Visualize single action"""
        viz_image = image.copy()
        
        # Draw single point
        cv2.circle(viz_image, 
                  (int(action.screen_x), int(action.screen_y)), 
                  10, (0, 255, 0), -1)
        
        # Add label
        cv2.putText(
            viz_image,
            f"Next: ({action.dx:.1f}, {action.dy:.1f}, {action.dz:.1f})",
            (int(action.screen_x) + 15, int(action.screen_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Save and display
        cv2.imwrite(f"{self.output_dir}/single_action_2d.jpg", cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
        
        # Create simple 3D visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot start and action
        ax.scatter([0], [0], [0], color='red', s=100, label='Start')
        ax.scatter([action.dx], [action.dy], [action.dz], 
                  color='green', s=100, label='Next Action')
        
        # Draw arrow
        ax.plot([0, action.dx], [0, action.dy], [0, action.dz], 
                '--', color='blue', alpha=0.5)
        
        # Customize plot
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Y (Forward/Back)')
        ax.set_zlabel('Z (Up/Down)')
        ax.set_title('Next Drone Action')
        ax.legend()
        ax.grid(True)
        ax.set_box_aspect([1,1,1])
        
        plt.savefig(f"{self.output_dir}/single_action_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
    '''
    def visualize_coordinate_system(self, image: np.ndarray = None) -> np.ndarray:
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

def test_drone_navigation():
    """Test the spatial understanding navigation system"""
    try:
        # Initialize
        image = cv2.imread('frame_1733321874.11946.jpg')
        if image is None:
            raise ValueError("Could not load test image")
        
        # Initialize controller
        controller = DroneController()
        instruction = "navigate through the center of the crane structure while avoiding obstacles"
        
        # Test both modes
        print("\n=== Testing Waypoint Mode ===")
        response = controller.process_spatial_command(image, instruction, mode="waypoint")
        print(response)
        
        print("\n=== Testing Single Action Mode ===")
        response = controller.process_spatial_command(image, instruction, mode="single")
        print(response)
        
    except KeyboardInterrupt:
        print("\nCaught Ctrl+C, exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.stop()

if __name__ == "__main__":
    test_drone_navigation() 