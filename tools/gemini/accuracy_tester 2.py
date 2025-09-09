#!/usr/bin/env python3
"""
Gemini Navigation Point Accuracy Testing Tool

This script tests Gemini's ability to generate accurate navigation points
from static images with various prompts. It helps identify which prompt
structures yield the most accurate and consistent results.
"""

import os
import sys
import cv2
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import argparse
import base64

# Add project root to path to allow importing from the parent directory
# This must be before any project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

# Import project modules
from action_projector import ActionProjector
from drone_space import ActionPoint

class GeminiAccuracyTester:
    """Test Gemini's accuracy in generating navigation points"""

    def __init__(self,
                output_dir: str = None,
                test_images_dir: str = None):
        """Initialize the tester with configuration options"""
        # Get project root directory
        self.project_root = project_root

        # Create output directory
        if output_dir is None:
            self.output_dir = os.path.join(self.project_root, "output", "gemini_tests")
        else:
            self.output_dir = output_dir

        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(self.output_dir, self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)

        # Set up test images directory
        if test_images_dir:
            self.test_images_dir = test_images_dir
        else:
            # Default to test_images in the tools/gemini directory
            self.test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")

        # Initialize action projector with correct Retina display dimensions
        self.action_projector = ActionProjector(
            image_width=3420,   # Actual captured width (2x scaling)
            image_height=2214   # Actual captured height (2x scaling)
        )

        # Print debug information
        print(f"ActionProjector configured with image dimensions: {self.action_projector.image_width}×{self.action_projector.image_height}")
        print(f"Using 2x scaling for Retina/HiDPI display")

        # Prompt variations to test
        self.prompt_variations = self._get_prompt_variations()

        # Test metrics
        self.metrics = {}

    def _get_prompt_variations(self) -> Dict[str, str]:
        """Define different prompt variations to test"""
        return {
            "baseline": "",  # Will use the default prompt in _get_single_action

            "Object-Centric":"""You are a drone navigation expert analyzing a drone camera view.

            Task: {instruction}

            First, identify ALL objects in the image that match the description "{instruction}".
            Then, select the MOST RELEVANT target object and place a single point DIRECTLY ON that object.

            Return in this exact JSON format:
            [{{"point": [y, x], "label": "action description"}}]

            Coordinate system:
            - x: 0-1000 scale (500=center, >500=right, <500=left)
            - y: 0-1000 scale (lower values=higher in image/sky)

            IMPORTANT:
            - Place the point PRECISELY on the target object, not nearby or in its direction
            - If the target is a vehicle or structure, aim for its center
            - Prioritize the clearest/closest matching object if multiple exist
            - Your accuracy in point placement is critical for navigation success""",

            "Two-Stage":"""You are a drone navigation expert analyzing a drone camera view.

            Task: {instruction}

            First, identify ALL objects in the image that match the description "{instruction}".
            Then, select the MOST RELEVANT target object and place a single point DIRECTLY ON that object.

            Return in this exact JSON format:
            [{{"point": [y, x], "label": "action description"}}]

            Coordinate system:
            - x: 0-1000 scale (500=center, >500=right, <500=left)
            - y: 0-1000 scale (lower values=higher in image/sky)

            IMPORTANT:
            - Place the point PRECISELY on the target object, not nearby or in its direction
            - If the target is a vehicle or structure, aim for its center
            - Prioritize the clearest/closest matching object if multiple exist
            - Your accuracy in point placement is critical for navigation success""",

            "Contextual Awareness":"""You are a drone navigation expert analyzing a 3D environment.

            Task: {instruction}

            This is a PIXEL-PERFECT targeting exercise. Your goal is to:

            1. Examine the entire image for ALL objects matching "{instruction}"
            2. For each matching object, assess:
            - Visibility (how clearly it can be seen)
            - Relevance to the instruction
            - Position in the frame
            3. Select the SINGLE most appropriate target object
            4. Place a point PRECISELY on the center of that object

            Return in this exact JSON format:
            [{{"point": [y, x], "label": "detailed description of the selected target"}}]

            Coordinate system:
            - x: 0-1000 scale (500=center, >500=right, <500=left)
            - y: 0-1000 scale (lower values=higher in image/sky)

            In your label, explicitly name the exact object you've targeted
            (e.g., "black police car," "tall palm tree," "checkered flag").""",

        }

    def run_tests(self,
                 tasks: List[str],
                 image_paths: List[str] = None):
        """Run accuracy tests on specified images with different prompts"""
        if not tasks:
            raise ValueError("No tasks specified. Please provide at least one task.")

        if image_paths is None:
            # Use all images in the test_images directory
            if os.path.exists(self.test_images_dir):
                image_paths = [
                    os.path.join(self.test_images_dir, f)
                    for f in os.listdir(self.test_images_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not os.path.isdir(os.path.join(self.test_images_dir, f))
                ]

                if not image_paths:
                    raise ValueError(f"No test images found in {self.test_images_dir}")
            else:
                raise ValueError(f"Test images directory {self.test_images_dir} does not exist")

        print(f"=== GEMINI NAVIGATION ACCURACY TEST ===")
        print(f"Testing {len(tasks)} tasks on {len(image_paths)} images with {len(self.prompt_variations)} prompt variations")
        print(f"Output directory: {self.results_dir}")
        print(f"Images directory: {self.test_images_dir}")

        # Track results
        results = []

        # For each image
        for img_path in image_paths:
            img_filename = os.path.basename(img_path)
            print(f"\nTesting image: {img_filename}")

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not load image {img_path}")
                continue

            # Convert to RGB (from BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # For each task
            for task in tasks:
                print(f"\n  Task: {task}")

                # Create visualization grid for this task - now with only 2 variations
                grid_size = len(self.prompt_variations)
                fig, axes = plt.subplots(1, grid_size, figsize=(20, 10))  # Larger size for high-res images
                if grid_size == 1:
                    axes = [axes]  # Make it iterable

                # For each prompt variation
                for i, (prompt_name, prompt_template) in enumerate(self.prompt_variations.items()):
                    print(f"    Testing {prompt_name} prompt...")

                    # Set up custom prompt if not baseline
                    if prompt_name != "baseline":
                        # Store original _get_single_action method
                        original_method = self.action_projector._get_single_action

                        # Create a modified method with custom prompt
                        def modified_get_single_action(image, instruction):
                            """Modified version with custom prompt"""
                            # IMPORTANT FIX: Convert RGB image back to BGR for proper OpenCV encoding
                            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            _, buffer = cv2.imencode('.jpg', image_bgr)
                            encoded_image = base64.b64encode(buffer).decode('utf-8')

                            # Format the prompt template
                            prompt = prompt_template.format(instruction=instruction)

                            # Continue with original implementation
                            try:
                                # Get response from Gemini
                                response = self.action_projector.model.generate_content([
                                    prompt,
                                    {'mime_type': 'image/jpeg', 'data': encoded_image}
                                ])

                                # Parse response text
                                response_text = response.text
                                if "```json" in response_text:
                                    json_start = response_text.find("```json") + 7
                                    json_end = response_text.find("```", json_start)
                                    response_text = response_text[json_start:json_end].strip()
                                elif "```" in response_text:
                                    json_start = response_text.find("```") + 3
                                    json_end = response_text.find("```", json_start)
                                    response_text = response_text[json_start:json_end].strip()

                                print(f"\n    Gemini Response for {prompt_name}:")
                                print(f"    {response_text}")

                                # Parse JSON response
                                points_data = json.loads(response_text)
                                if not points_data:
                                    raise ValueError("No points returned from Gemini")

                                # Take first (and should be only) point
                                point_info = points_data[0]

                                # Convert normalized coordinates to pixel coordinates
                                y, x = point_info['point']
                                pixel_x = int((x / 1000.0) * self.action_projector.image_width)
                                pixel_y = int((y / 1000.0) * self.action_projector.image_height)

                                # Project 2D point to 3D
                                x3d, y3d, z3d = self.action_projector.reverse_project_point((pixel_x, pixel_y))

                                # Create ActionPoint
                                action = ActionPoint(
                                    dx=x3d, dy=y3d, dz=z3d,
                                    action_type="move",
                                    screen_x=pixel_x,
                                    screen_y=pixel_y
                                )

                                return action

                            except Exception as e:
                                print(f"    Error in custom prompt test: {e}")
                                print("    Full response:")
                                print(response.text)
                                return None

                        # Replace method temporarily
                        self.action_projector._get_single_action = modified_get_single_action

                    # Run the test with this prompt variation
                    start_time = time.time()
                    action = None

                    try:
                        # Use the action_projector to get a single point
                        self.action_projector.set_mode("single")
                        actions = self.action_projector.get_vlm_points(image, task)

                        if actions and actions[0]:
                            action = actions[0]

                            # Store result data
                            result = {
                                "image": img_filename,
                                "task": task,
                                "prompt": prompt_name,
                                "point_2d": [int(action.screen_x), int(action.screen_y)],
                                "point_3d": [float(action.dx), float(action.dy), float(action.dz)],
                                "processing_time": time.time() - start_time,
                            }
                            results.append(result)

                            # Create visualization
                            ax = axes[i]
                            viz_img = image.copy()

                            # Get the original pixel coordinates directly
                            px, py = int(action.screen_x), int(action.screen_y)

                            # Draw point at the exact pixel coordinates
                            cv2.circle(viz_img, (px, py), 15, (0, 255, 0), -1)

                            # Draw vector from center
                            center = (viz_img.shape[1]//2, viz_img.shape[0]//2)
                            cv2.line(viz_img, center, (px, py), (0, 255, 255), 2)

                            # Add label
                            cv2.putText(viz_img,
                                f"Pixel: ({px}, {py})",
                                (px + 20, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                            # Add 3D vector label
                            cv2.putText(viz_img,
                                f"3D: ({action.dx:.1f}, {action.dy:.1f}, {action.dz:.1f})",
                                (px + 20, py + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                            # Add normalized coordinates label for verification
                            # Converting to the 0-1000 scale expected by Gemini
                            norm_x = int((px / self.action_projector.image_width) * 1000)
                            norm_y = int((py / self.action_projector.image_height) * 1000)
                            cv2.putText(viz_img,
                                f"Norm: ({norm_y}, {norm_x}) [0-1000 scale]",
                                (px + 20, py + 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                            # Save individual visualization for this task and prompt
                            indiv_path = os.path.join(
                                self.results_dir,
                                f"{img_filename.split('.')[0]}_{task.replace(' ', '_')}_{prompt_name}.jpg"
                            )
                            cv2.imwrite(indiv_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))

                            # Display in the grid
                            ax.imshow(viz_img)
                            ax.set_title(f"{prompt_name}\nPixel: ({px}, {py})\nNorm: ({norm_y}, {norm_x})", fontsize=10)
                            ax.axis('off')
                    except Exception as e:
                        print(f"    Error in test: {e}")
                        # Add empty plot
                        axes[i].imshow(image)
                        axes[i].set_title(f"{prompt_name} (ERROR)")
                        axes[i].axis('off')

                    # Restore original method if we modified it
                    if prompt_name != "baseline":
                        self.action_projector._get_single_action = original_method

                # Save the comparison grid
                task_slug = task.replace(" ", "_").lower()
                img_name = os.path.splitext(img_filename)[0]
                grid_filename = f"{img_name}_{task_slug}_comparison.png"
                plt.suptitle(f"Task: {task}", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, grid_filename), dpi=150)
                plt.close()

                print(f"  Saved comparison to {grid_filename}")

        # Save all results to JSON
        results_file = os.path.join(self.results_dir, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nTest complete. Results saved to {self.results_dir}")
        return results

    def analyze_results(self, results):
        """Analyze test results and generate metrics"""
        if not results:
            print("No results to analyze")
            return

        # Group by prompt variation
        prompt_results = {}
        for result in results:
            prompt = result["prompt"]
            if prompt not in prompt_results:
                prompt_results[prompt] = []
            prompt_results[prompt].append(result)

        # Calculate metrics for each prompt
        metrics = {}
        for prompt, results in prompt_results.items():
            metrics[prompt] = {
                "count": len(results),
                "avg_processing_time": sum(r["processing_time"] for r in results) / len(results),
                # Calculate center distance (how far points are from image center on average)
                "avg_center_distance": sum(
                    np.sqrt(
                        (r["point_2d"][0] - self.action_projector.image_width/2)**2 +
                        (r["point_2d"][1] - self.action_projector.image_height/2)**2
                    ) for r in results
                ) / len(results),
                # Other metrics can be added here
            }

        # Print metrics
        print("\n=== PROMPT PERFORMANCE METRICS ===")
        for prompt, m in metrics.items():
            print(f"\n{prompt.upper()} PROMPT:")
            print(f"  Test count: {m['count']}")
            print(f"  Avg processing time: {m['avg_processing_time']:.2f} sec")
            print(f"  Avg distance from center: {m['avg_center_distance']:.2f} pixels")

        # Create metrics visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Processing time comparison
        prompts = list(metrics.keys())
        times = [metrics[p]["avg_processing_time"] for p in prompts]
        ax1.bar(prompts, times)
        ax1.set_title("Average Processing Time (sec)")
        ax1.set_ylabel("Seconds")
        ax1.set_xticklabels(prompts, rotation=45, ha="right")

        # Center distance comparison
        distances = [metrics[p]["avg_center_distance"] for p in prompts]
        ax2.bar(prompts, distances)
        ax2.set_title("Average Distance from Center (pixels)")
        ax2.set_ylabel("Pixels")
        ax2.set_xticklabels(prompts, rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "metrics_comparison.png"), dpi=150)
        plt.close()

        # Save metrics
        metrics_file = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.metrics = metrics
        return metrics

def main():
    """Main entry point for the Gemini accuracy tester"""
    parser = argparse.ArgumentParser(description="Test Gemini's navigation point accuracy")
    parser.add_argument("--images", type=str, default=None, help="Directory with test images")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--tasks", type=str, nargs="+", help="Tasks to test (if not specified, default tasks will be used)")
    args = parser.parse_args()

    # Initialize the tester
    tester = GeminiAccuracyTester(
        output_dir=args.output,
        test_images_dir=args.images
    )

    # Run the tests
    results = tester.run_tests(
        tasks=args.tasks
    )

    # Analyze results
    tester.analyze_results(results)

    print("\nTest process complete!")
    print(f"Review the results in {tester.results_dir}")

if __name__ == "__main__":
    main()
