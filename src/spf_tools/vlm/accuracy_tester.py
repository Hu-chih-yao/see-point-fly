#!/usr/bin/env python3
"""
VLM Navigation Point Accuracy Testing Tool for SPF Framework

This module provides comprehensive testing capabilities for VLM (Vision Language Model)
navigation point generation accuracy. It supports multiple VLM providers through the
unified VLMClient interface and evaluates different prompt strategies.

The tool helps identify optimal prompt structures and VLM configurations for
accurate navigation point generation in drone applications.
"""

import os
import sys
import cv2
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import argparse

# Import SPF modules
try:
    from spf.clients.vlm_client import VLMClient
    from spf.base.drone_space import ActionPoint
    from spf_tools.capture import capture_screen, prepare_for_vlm
except ImportError as e:
    print(f"Error importing SPF modules: {e}")
    print("Make sure SPF package is properly installed and in Python path")
    sys.exit(1)


class VLMAccuracyTester:
    """
    Test VLM accuracy for navigation point generation across different providers and prompts.
    """

    def __init__(self,
                 output_dir: Optional[str] = None,
                 test_images_dir: Optional[str] = None,
                 image_width: int = 1920,
                 image_height: int = 1080):
        """
        Initialize the VLM accuracy tester.

        Args:
            output_dir: Directory to save test results
            test_images_dir: Directory containing test images
            image_width: Expected image width for coordinate normalization
            image_height: Expected image height for coordinate normalization
        """
        # Setup directories
        if output_dir is None:
            self.output_dir = Path("output/vlm_tests")
        else:
            self.output_dir = Path(output_dir)

        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.output_dir / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if test_images_dir:
            self.test_images_dir = Path(test_images_dir)
        else:
            # Default to test_images in the vlm directory
            self.test_images_dir = Path(__file__).parent / "test_images"

        # Image dimensions for coordinate conversion
        self.image_width = image_width
        self.image_height = image_height

        print(f"VLM Accuracy Tester initialized")
        print(f"  Image dimensions: {self.image_width}x{self.image_height}")
        print(f"  Test images: {self.test_images_dir}")
        print(f"  Output directory: {self.results_dir}")

        # Define prompt variations to test
        self.prompt_variations = self._get_prompt_variations()

        # Store test results
        self.test_results = []

    def _get_prompt_variations(self) -> Dict[str, str]:
        """
        Define different prompt variations for testing VLM accuracy.

        Returns:
            Dictionary mapping prompt names to prompt templates
        """
        return {
            "baseline": """You are a drone navigation expert analyzing a drone camera view.

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
- Your accuracy in point placement is critical for navigation success""",

            "precise": """You are a drone navigation expert performing PIXEL-PERFECT target identification.

Task: {instruction}

ANALYSIS PROCESS:
1. Scan the entire image systematically for objects matching "{instruction}"
2. For each candidate, evaluate:
   - Object clarity and visibility
   - Relevance to the navigation task
   - Size and prominence in frame
3. Select the SINGLE most appropriate target
4. Place point at the exact center of the selected object

Return in this exact JSON format:
[{{"point": [y, x], "depth": depth_value, "label": "detailed target description"}}]

Coordinate system:
- x: 0-1000 scale (500=center, >500=right, <500=left)
- y: 0-1000 scale (lower values=higher in image/sky)
- depth: 1-10 scale (1=very close, 10=far away)

CRITICAL REQUIREMENTS:
- Point must be PRECISELY on the target object, not nearby
- Label must specifically describe the chosen target
- Depth must reflect object's apparent distance/size in frame""",

            "contextual": """You are an expert drone pilot analyzing a flight camera view for navigation.

Mission: {instruction}

STEP-BY-STEP ANALYSIS:
1. IDENTIFY: Locate all objects matching "{instruction}" in the image
2. PRIORITIZE: Rank them by:
   - Navigation relevance and safety
   - Object clarity and size
   - Accessibility for drone approach
3. TARGET: Select the optimal target for the mission
4. MARK: Place navigation point at target's geometric center

Return in this exact JSON format:
[{{"point": [y, x], "depth": depth_value, "label": "mission-specific target description"}}]

Coordinate reference:
- x: 0-1000 (500=image center, 0=left edge, 1000=right edge)
- y: 0-1000 (0=top edge, 500=middle, 1000=bottom edge)
- depth: 1-10 (1=foreground/large, 10=background/small)

NAVIGATION PRECISION:
- Target point MUST be on the object, not in its direction
- Consider flight safety and approach angles
- Provide specific target identification in label"""
        }

    def test_vlm_provider(self,
                         provider: str,
                         model: str,
                         image: np.ndarray,
                         instruction: str,
                         prompt_name: str) -> Optional[Dict]:
        """
        Test a specific VLM provider with given image and instruction.

        Args:
            provider: VLM provider name (e.g., "gemini", "openai")
            model: Model name to use
            image: Input image as numpy array (RGB)
            instruction: Navigation instruction
            prompt_name: Name of the prompt variation being tested

        Returns:
            Dictionary containing test results or None if test failed
        """
        try:
            # Initialize VLM client
            vlm_client = VLMClient(provider, model)

            # Get prompt template
            prompt_template = self.prompt_variations[prompt_name]
            prompt = prompt_template.format(instruction=instruction)

            # Measure response time
            start_time = time.time()

            # Get VLM response
            response_text = vlm_client.generate_response(prompt, image)

            # Clean and parse response
            response_text = VLMClient.clean_response_text(response_text)
            processing_time = time.time() - start_time

            print(f"  {provider.upper()} ({prompt_name}) Response:")
            print(f"  {response_text}")

            # Parse JSON response
            try:
                points_data = json.loads(response_text)
                if not points_data:
                    raise ValueError("No points returned from VLM")

                # Extract first point
                point_info = points_data[0]
                y, x = point_info['point']
                depth = point_info.get('depth', 5)
                label = point_info.get('label', 'target')

                # Convert normalized coordinates to pixel coordinates
                pixel_x = int((x / 1000.0) * self.image_width)
                pixel_y = int((y / 1000.0) * self.image_height)

                # Ensure coordinates are within bounds
                pixel_x = max(0, min(pixel_x, self.image_width - 1))
                pixel_y = max(0, min(pixel_y, self.image_height - 1))

                result = {
                    "provider": provider,
                    "model": model,
                    "prompt": prompt_name,
                    "instruction": instruction,
                    "processing_time": processing_time,
                    "point_normalized": [y, x],
                    "point_pixel": [pixel_x, pixel_y],
                    "depth": depth,
                    "label": label,
                    "success": True,
                    "raw_response": response_text
                }

                print(f"  ✅ Success: Point at ({pixel_x}, {pixel_y}), depth {depth}")
                return result

            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing failed: {e}")
                return {
                    "provider": provider,
                    "model": model,
                    "prompt": prompt_name,
                    "instruction": instruction,
                    "processing_time": processing_time,
                    "success": False,
                    "error": f"JSON parsing error: {e}",
                    "raw_response": response_text
                }

        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            return {
                "provider": provider,
                "model": model,
                "prompt": prompt_name,
                "instruction": instruction,
                "success": False,
                "error": str(e)
            }

    def run_accuracy_tests(self,
                          providers: List[Tuple[str, str]],
                          instructions: List[str],
                          image_paths: Optional[List[str]] = None) -> List[Dict]:
        """
        Run comprehensive accuracy tests across providers, prompts, and images.

        Args:
            providers: List of (provider_name, model_name) tuples
            instructions: List of navigation instructions to test
            image_paths: Optional list of image paths (uses test_images_dir if None)

        Returns:
            List of test result dictionaries
        """
        if image_paths is None:
            # Use all images in test_images directory
            if self.test_images_dir.exists():
                image_paths = [
                    str(p) for p in self.test_images_dir.glob("*.jpg")
                    if p.is_file()
                ] + [
                    str(p) for p in self.test_images_dir.glob("*.png")
                    if p.is_file()
                ]
            else:
                raise ValueError(f"Test images directory {self.test_images_dir} does not exist")

        if not image_paths:
            raise ValueError("No test images found")

        print(f"=== VLM ACCURACY TEST SUITE ===")
        print(f"Providers: {len(providers)}")
        print(f"Prompts: {len(self.prompt_variations)}")
        print(f"Instructions: {len(instructions)}")
        print(f"Images: {len(image_paths)}")
        print(f"Total tests: {len(providers) * len(self.prompt_variations) * len(instructions) * len(image_paths)}")

        results = []

        # Test each combination
        for img_path in image_paths:
            img_name = Path(img_path).stem
            print(f"\n📸 Testing image: {img_name}")

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ❌ Could not load image: {img_path}")
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image to expected dimensions if needed
            if image.shape[1] != self.image_width or image.shape[0] != self.image_height:
                image = cv2.resize(image, (self.image_width, self.image_height))

            for instruction in instructions:
                print(f"\n  🎯 Instruction: {instruction}")

                # Create comparison visualization
                n_tests = len(providers) * len(self.prompt_variations)
                fig, axes = plt.subplots(1, min(n_tests, 6), figsize=(20, 6))
                if n_tests == 1:
                    axes = [axes]
                elif n_tests > 6:
                    # If too many tests, just show first 6
                    axes = axes[:6]

                test_idx = 0

                for provider, model in providers:
                    for prompt_name in self.prompt_variations:
                        print(f"\n    🤖 Testing {provider}:{model} with {prompt_name} prompt...")

                        result = self.test_vlm_provider(
                            provider, model, image, instruction, prompt_name
                        )

                        if result:
                            result["image_name"] = img_name
                            results.append(result)

                            # Add to visualization if space available
                            if test_idx < len(axes) and result["success"]:
                                ax = axes[test_idx]
                                viz_img = image.copy()

                                # Draw point
                                px, py = result["point_pixel"]
                                cv2.circle(viz_img, (px, py), 8, (0, 255, 0), -1)

                                # Add label
                                cv2.putText(viz_img, f"{result['label'][:20]}...",
                                          (px + 10, py - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                                ax.imshow(viz_img)
                                ax.set_title(f"{provider}/{prompt_name}\n({px}, {py})", fontsize=8)
                                ax.axis('off')

                        test_idx += 1

                # Save comparison visualization
                if results:
                    comparison_path = self.results_dir / f"{img_name}_{instruction.replace(' ', '_')}_comparison.png"
                    plt.suptitle(f"Instruction: {instruction}", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                    plt.close()

        # Save results to JSON
        results_file = self.results_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Testing complete! {len(results)} tests run")
        print(f"Results saved to: {results_file}")

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """
        Analyze test results and generate performance metrics.

        Args:
            results: List of test result dictionaries

        Returns:
            Dictionary containing analysis metrics
        """
        if not results:
            print("No results to analyze")
            return {}

        print("\n=== RESULT ANALYSIS ===")

        # Group results by provider and prompt
        analysis = {}

        for result in results:
            if not result.get("success", False):
                continue

            key = f"{result['provider']}/{result['prompt']}"

            if key not in analysis:
                analysis[key] = {
                    "tests": [],
                    "success_count": 0,
                    "avg_processing_time": 0,
                    "avg_center_distance": 0
                }

            analysis[key]["tests"].append(result)
            analysis[key]["success_count"] += 1

        # Calculate metrics for each group
        for key, data in analysis.items():
            tests = data["tests"]

            # Average processing time
            data["avg_processing_time"] = sum(t["processing_time"] for t in tests) / len(tests)

            # Average distance from center
            center_distances = []
            for test in tests:
                px, py = test["point_pixel"]
                center_x, center_y = self.image_width // 2, self.image_height // 2
                distance = np.sqrt((px - center_x)**2 + (py - center_y)**2)
                center_distances.append(distance)

            data["avg_center_distance"] = sum(center_distances) / len(center_distances)

        # Print summary
        print("\nPERFORMACE METRICS:")
        for key, metrics in analysis.items():
            print(f"\n{key}:")
            print(f"  Successful tests: {metrics['success_count']}")
            print(f"  Avg processing time: {metrics['avg_processing_time']:.2f}s")
            print(f"  Avg center distance: {metrics['avg_center_distance']:.1f}px")

        # Create metrics visualization
        if len(analysis) > 1:
            self._create_metrics_visualization(analysis)

        # Save analysis
        analysis_file = self.results_dir / "analysis.json"
        with open(analysis_file, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_analysis = {}
            for key, data in analysis.items():
                serializable_analysis[key] = {
                    "success_count": data["success_count"],
                    "avg_processing_time": float(data["avg_processing_time"]),
                    "avg_center_distance": float(data["avg_center_distance"])
                }
            json.dump(serializable_analysis, f, indent=2)

        return analysis

    def _create_metrics_visualization(self, analysis: Dict) -> None:
        """Create visualization comparing metrics across providers/prompts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        keys = list(analysis.keys())
        processing_times = [analysis[k]["avg_processing_time"] for k in keys]
        center_distances = [analysis[k]["avg_center_distance"] for k in keys]

        # Processing time comparison
        ax1.bar(range(len(keys)), processing_times)
        ax1.set_title("Average Processing Time")
        ax1.set_ylabel("Seconds")
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels([k.replace("/", "\n") for k in keys], rotation=45, ha="right")

        # Center distance comparison
        ax2.bar(range(len(keys)), center_distances)
        ax2.set_title("Average Distance from Center")
        ax2.set_ylabel("Pixels")
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels([k.replace("/", "\n") for k in keys], rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(self.results_dir / "metrics_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for the VLM accuracy tester."""
    parser = argparse.ArgumentParser(description="Test VLM navigation point accuracy")
    parser.add_argument("--instructions", type=str, nargs="+", required=True,
                       help="Navigation instructions to test")
    parser.add_argument("--providers", type=str, nargs="+",
                       default=["gemini:gemini-2.5-flash", "openai:openai/gpt-4.1"],
                       help="Provider:model pairs to test")
    parser.add_argument("--images", type=str, help="Directory with test images")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")

    args = parser.parse_args()

    # Parse provider:model pairs
    providers = []
    for provider_spec in args.providers:
        if ":" in provider_spec:
            provider, model = provider_spec.split(":", 1)
        else:
            # Default models
            if provider_spec == "gemini":
                provider, model = "gemini", "gemini-2.5-flash"
            elif provider_spec == "openai":
                provider, model = "openai", "openai/gpt-4.1"
            else:
                print(f"Warning: Unknown provider {provider_spec}, skipping")
                continue
        providers.append((provider, model))

    # Initialize tester
    tester = VLMAccuracyTester(
        output_dir=args.output,
        test_images_dir=args.images,
        image_width=args.width,
        image_height=args.height
    )

    # Run tests
    results = tester.run_accuracy_tests(providers, args.instructions)

    # Analyze results
    tester.analyze_results(results)

    print(f"\n✅ Testing complete! Check results in {tester.results_dir}")


if __name__ == "__main__":
    main()
