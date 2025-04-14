#!/usr/bin/env python3
"""
Wrapper script to run the Gemini accuracy tester with proper path handling.
This script evaluates Gemini's navigation point accuracy using images
from the test_images directory with specified tasks.
"""

import os
import sys
import argparse

# Determine the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# Add project root to path
sys.path.insert(0, project_root)

# Import the accuracy tester
from tools.gemini.accuracy_tester import GeminiAccuracyTester

def main():
    """Entry point for the accuracy test wrapper"""
    parser = argparse.ArgumentParser(description="Test Gemini's navigation point accuracy")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                       help="Tasks to test (required, e.g., 'fly toward the car')")
    args = parser.parse_args()
    
    # Set fixed directories
    test_images_dir = os.path.join(script_dir, "test_images")
    output_dir = os.path.join(project_root, "output", "gemini_tests")
    
    print(f"Using test images from: {test_images_dir}")
    print(f"Saving results to: {output_dir}")
    
    # Initialize the tester with fixed paths
    tester = GeminiAccuracyTester(
        output_dir=output_dir,
        test_images_dir=test_images_dir
    )
    
    # Run the tests with required tasks
    results = tester.run_tests(tasks=args.tasks)
    
    # Analyze results
    tester.analyze_results(results)
    
    print("\nTest process complete!")
    print(f"Review the results in {tester.results_dir}")

if __name__ == "__main__":
    main() 