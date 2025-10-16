#!/usr/bin/env python3
"""
AirSim main module for SPF (See, Point, Fly)
Uses AirSim API for image capture, depth estimation, and LLM-based command processing
"""

import os
import time
import cv2
import numpy as np
import airsim

from .controller import AirSimController


def check_camera_settings(client, camera_name="0"):
    """Check and display camera settings, provide guidance if resolution is too low

    Args:
        client: AirSim MultirotorClient instance
        camera_name: Camera name/ID

    Returns:
        tuple: (width, height, needs_config)
    """
    try:
        camera_info = client.simGetCameraInfo(camera_name)
        print(f"\n=== CAMERA SETTINGS ===")
        print(f"Camera: {camera_name}")
        print(f"FOV: {camera_info.fov} degrees")
        print(f"Pose: {camera_info.pose}")

        responses = client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        )

        if responses and responses[0].width > 0:
            width = responses[0].width
            height = responses[0].height
            print(f"Resolution: {width}x{height}")

            if width < 640 or height < 480:
                print(f"\n WARNING: Camera resolution is very low ({width}x{height})")
                print(
                    f" Default AirSim camera is 256x144, which is too small for good navigation"
                )
                print(f"\n To fix this, configure AirSim settings.json:")
                print(
                    f"   https://microsoft.github.io/AirSim/settings/#where-are-settings-stored"
                )
                print(
                    f"\n A sample settings.json is provided in 'src/spf/airsim/settings.json.example'"
                )
                print(
                    f"\n Copy settings.json.example to the appropriate location and rename it to settings.json"
                )
                print(f"=" * 50)
                return width, height, True
            else:
                print(f"✓ Camera resolution is acceptable")
                print(f"=" * 50)
                return width, height, False
        else:
            print(f"  Could not determine camera resolution")
            return 640, 480, True

    except Exception as e:
        print(f"Error checking camera settings: {e}")
        return 640, 480, True


def capture_airsim_image(client, camera_name="0"):
    """Capture image from AirSim using API

    Args:
        client: AirSim MultirotorClient instance
        camera_name: Camera name/ID to capture from

    Returns:
        numpy array: Captured image in BGR format
    """
    try:
        responses = client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        )

        if not responses:
            print("Error: No image response from AirSim")
            return None

        response = responses[0]

        if response.height == 0 or response.width == 0:
            print(f"Error: Invalid image dimensions {response.width}x{response.height}")
            return None

        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_bgr = img1d.reshape(response.height, response.width, 3)

        return img_bgr

    except Exception as e:
        print(f"Error capturing AirSim image: {e}")
        return None

        return 640, 480


def main(args):
    """Main entrypoint for AirSim mode"""

    if args.test:
        print("\n=== TEST MODE WITH STATIC IMAGE ===")
        test_image_path = "frame_airsim_test.jpg"

        if not os.path.exists(test_image_path):
            print(f"Error: Test image '{test_image_path}' not found")
            return 1

        test_image = cv2.imread(test_image_path)

        try:
            import yaml

            with open("config_airsim.yaml", "r") as f:
                config = yaml.safe_load(f)
                adaptive_mode = config.get("adaptive_mode", True)
        except Exception as e:
            print(f"Config loading failed, using default adaptive mode: {e}")
            adaptive_mode = True

        image_height, image_width = test_image.shape[:2]
        print(f"Using test image dimensions: {image_width}x{image_height}")

        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)

        controller = AirSimController(
            adaptive_mode=adaptive_mode,
            image_width=image_width,
            image_height=image_height,
            config=config,
        )

        instruction = "navigate through the crane structure safely"

        response = controller.process_spatial_command(test_image, instruction)
        print(f"\nAction Response:\n{response}\n")

        controller.stop()
        return 0

    print("\n=== STARTING DRONE SPATIAL NAVIGATION (AIRSIM) ===")

    try:
        import yaml

        with open("config_airsim.yaml", "r") as f:
            config = yaml.safe_load(f)
            adaptive_mode = config.get("adaptive_mode", True)
            camera_name = config.get("camera_name", "0")
            command_loop_delay = config.get("command_loop_delay", 0)
            print(f"Adaptive Mode: {adaptive_mode}")
            print(f"Camera Name: {camera_name}")
            print(f"Command loop delay: {command_loop_delay}s")
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        command_loop_delay = 0
        adaptive_mode = True
        camera_name = "0"

    airsim_controller = None
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("Connected to AirSim")

        image_width, image_height, needs_config = check_camera_settings(
            client, camera_name
        )

        if needs_config:
            response = input("\nCamera resolution is low. Continue anyway? (y/n): ")
            if response.lower() != "y":
                print("Please update settings.json and restart AirSim, then try again.")
                return 1

        airsim_controller = AirSimController(
            adaptive_mode=adaptive_mode,
            image_width=image_width,
            image_height=image_height,
            config=config,
        )

        current_command = input(
            "\nEnter high-level command (e.g., 'navigate through the center of the crane structure'): "
        )

        print("Starting in 3 seconds...")
        time.sleep(3)

        airsim_controller.takeoff()

        print("\nStarting control loop...")
        print("Press Ctrl+C to exit")

        while True:
            if args.debug:
                print("Waiting for previous actions to complete...")
                airsim_controller.wait_for_queue_empty(debug=True)
                print("Action queue empty, processing new frame...")
            else:
                airsim_controller.wait_for_queue_empty()

            frame = capture_airsim_image(client, camera_name)

            if frame is None:
                print("Error: Failed to capture image from AirSim")
                time.sleep(1)
                continue

            response = airsim_controller.process_spatial_command(frame, current_command)
            print(f"\nAction Response:\n{response}\n")

            time.sleep(command_loop_delay)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if airsim_controller is not None:
            airsim_controller.stop()

    return 0
