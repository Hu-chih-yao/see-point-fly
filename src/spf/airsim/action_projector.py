import cv2
import numpy as np
from ..base.action_projector import ActionProjector
from ..base.drone_space import ActionPoint
from .drone_space import AirSimDroneActionSpace
from typing import List, Tuple
import os
import time
import json


class AirSimActionProjector(ActionProjector):
    def __init__(
        self,
        image_width=1920,
        image_height=1080,
        adaptive_mode=False,
        config_path="config_airsim.yaml",
    ):
        super().__init__(image_width, image_height, config_path)

        self.adaptive_mode = adaptive_mode

        self.action_space = AirSimDroneActionSpace(n_samples=8)

        print(
            f"[AirSimActionProjector] Initialized with {self.api_provider} provider using model: {self.model_name}"
        )

    def _determine_model_name(self):
        if self.custom_model:
            return self.custom_model

        if self.api_provider == "openai":
            return "google/gemini-2.5-flash"
        else:
            return "gemini-2.5-flash"

    def get_vlm_points(
        self, image: np.ndarray, instruction: str, **kwargs
    ) -> List[ActionPoint]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            actions = [self._get_single_action(image, instruction)]

            if actions and actions[0] is not None:
                viz_image = image.copy()

                for i, action in enumerate(actions, 1):
                    cv2.circle(
                        viz_image,
                        (int(action.screen_x), int(action.screen_y)),
                        10,
                        (0, 255, 0),
                        -1,
                    )

                    cv2.putText(
                        viz_image,
                        f"{i}: ({action.dx:.1f}, {action.dy:.1f}, {action.dz:.1f})",
                        (int(action.screen_x) + 15, int(action.screen_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                save_path = f"{self.output_dir}/decision_{timestamp}.jpg"
                cv2.imwrite(save_path, viz_image)

                decision_data = {
                    "timestamp": timestamp,
                    "mode": "airsim",
                    "adaptive_mode": self.adaptive_mode,
                    "instruction": instruction,
                    "actions": [],
                }

                for action in actions:
                    action_data = {
                        "dx": action.dx,
                        "dy": action.dy,
                        "dz": action.dz,
                        "screen_x": action.screen_x,
                        "screen_y": action.screen_y,
                    }

                    if (
                        hasattr(action, "adaptive_depth")
                        and action.adaptive_depth is not None
                    ):
                        action_data["adaptive_depth"] = action.adaptive_depth
                    if hasattr(action, "vlm_depth") and action.vlm_depth is not None:
                        action_data["vlm_depth"] = action.vlm_depth

                    decision_data["actions"].append(action_data)

                with open(f"{self.output_dir}/decision_{timestamp}.json", "w") as f:
                    json.dump(decision_data, f, indent=2)

            return actions

        except Exception as e:
            print(f"Error getting points: {e}")
            return []

    def _get_single_action(
        self, image: np.ndarray, instruction: str, **kwargs
    ) -> ActionPoint:
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
            response_text = self.vlm_client.generate_response(prompt, image)

            from ..clients.vlm_client import VLMClient

            response_text = VLMClient.clean_response_text(response_text)

            print(f"\n{self.api_provider.upper()} Response:")
            print(response_text)

            points_data = json.loads(response_text)
            if not points_data:
                raise ValueError("No points returned from VLM")

            point_info = points_data[0]

            y, x = point_info["point"]
            pixel_x = int((x / 1000.0) * self.image_width)
            pixel_y = int((y / 1000.0) * self.image_height)

            vlm_depth = point_info.get("depth", 4)

            if self.adaptive_mode:
                adaptive_depth = self._calculate_adaptive_depth(vlm_depth)
                depth_for_projection = adaptive_depth
            else:
                adaptive_depth = None
                depth_for_projection = vlm_depth / 10.0 * 2.0

            x3d, y3d, z3d = self.reverse_project_point(
                (pixel_x, pixel_y), depth=depth_for_projection
            )

            action = ActionPoint(
                dx=x3d,
                dy=y3d,
                dz=z3d,
                action_type="move",
                screen_x=pixel_x,
                screen_y=pixel_y,
            )

            if self.adaptive_mode:
                action.adaptive_depth = adaptive_depth
                action.vlm_depth = vlm_depth

            print(f"\nIdentified single action: {point_info['label']}")
            print(f"2D Normalized: ({x}, {y})")
            print(f"2D Pixels: ({pixel_x}, {pixel_y})")
            print(f"VLM Depth: {vlm_depth}/10")
            if self.adaptive_mode:
                print(f"Adaptive Depth: {adaptive_depth:.2f}")
            print(f"3D Vector: ({x3d:.2f}, {y3d:.2f}, {z3d:.2f})")

            return action

        except Exception as e:
            print(f"Error in single action mode: {e}")
            print("Full response:")
            if "response_text" in locals():
                print(response_text)
            return None

    def _calculate_adaptive_depth(self, vlm_depth):
        if vlm_depth <= 2:
            adaptive_depth = 0
            print(
                f"AirSim: VLM depth {vlm_depth}/10 → Adaptive depth {adaptive_depth} (No movement - too close)"
            )
        elif vlm_depth <= 5:
            adaptive_depth = (vlm_depth / 10.0) * 2
            print(
                f"AirSim: VLM depth {vlm_depth}/10 → Adaptive depth {adaptive_depth:.2f} (Careful movement)"
            )
        else:
            adaptive_depth = 1.0 + (vlm_depth - 5) / 5.0
            print(
                f"AirSim: VLM depth {vlm_depth}/10 → Adaptive depth {adaptive_depth:.2f} (Normal movement)"
            )

        return adaptive_depth
