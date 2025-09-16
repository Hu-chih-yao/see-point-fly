#!/usr/bin/env python3
"""
Comprehensive System Diagnostics for SPF Framework

This module provides comprehensive system checks for the SPF drone navigation system.
Tests monitor configuration, screen capture, VLM client functionality, and processing pipeline.
"""

import os
import sys
import cv2
import numpy as np
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Import SPF modules
try:
    from spf.clients.vlm_client import VLMClient
    from spf.tello.action_projector import TelloActionProjector
    from spf.sim.action_projector import SimActionProjector
    from spf_tools.capture import capture_screen, capture_screen_resized, prepare_for_vlm, get_monitor_info
except ImportError as e:
    print(f"Warning: Could not import SPF modules: {e}")
    print("Some diagnostic features may not be available.")


def check_monitors() -> bool:
    """
    Check monitor configuration and scaling detection.

    Returns:
        True if monitors are properly detected, False otherwise
    """
    print("\n=== MONITOR CONFIGURATION CHECK ===")

    try:
        monitor_info = get_monitor_info()

        if not monitor_info:
            print("❌ No monitors detected")
            return False

        print(f"✅ Detected {len(monitor_info)} monitor(s)")

        for monitor_name, info in monitor_info.items():
            print(f"\n{monitor_name.upper()}:")
            print(f"  Logical dimensions: {info['width']}x{info['height']}")
            print(f"  Captured dimensions: {info['captured_width']}x{info['captured_height']}")
            print(f"  Scaling factor: {info['width_scaling']:.2f}x")

            if info['is_hidpi']:
                print(f"  📱 HiDPI/Retina display detected")
            else:
                print(f"  🖥️  Standard display")

        return True

    except Exception as e:
        print(f"❌ Monitor check failed: {e}")
        return False


def check_capture() -> bool:
    """
    Test screen capture functionality with both regular and resized methods.

    Returns:
        True if capture works correctly, False otherwise
    """
    print("\n=== SCREEN CAPTURE TEST ===")

    # Create output directory
    output_path = Path("output/diagnostics")
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        monitor_info = get_monitor_info()
        if not monitor_info:
            print("❌ No monitor info available for capture test")
            return False

        monitor = monitor_info["monitor_1"]

        # Test regular capture
        print("\nTesting regular capture (full resolution):")
        regular_capture = capture_screen(monitor_index=1)
        print(f"  Captured dimensions: {regular_capture.shape[1]}x{regular_capture.shape[0]}")

        # Save regular capture
        cv2.imwrite(
            str(output_path / "capture_regular.jpg"),
            cv2.cvtColor(regular_capture, cv2.COLOR_RGB2BGR)
        )

        # Test resized capture
        print("\nTesting resized capture (matches logical resolution):")
        resized_capture = capture_screen_resized(monitor_index=1)
        print(f"  Captured dimensions: {resized_capture.shape[1]}x{resized_capture.shape[0]}")

        # Save resized capture
        cv2.imwrite(
            str(output_path / "capture_resized.jpg"),
            cv2.cvtColor(resized_capture, cv2.COLOR_RGB2BGR)
        )

        # Verify resized capture matches logical dimensions
        if (resized_capture.shape[1] == monitor['width'] and
            resized_capture.shape[0] == monitor['height']):
            print("✅ Resized capture matches logical dimensions")
        else:
            print("❌ Resized capture does not match logical dimensions")

        # Test scaling detection
        scaling_factor = regular_capture.shape[1] / monitor['width']
        print(f"Detected scaling factor: {scaling_factor:.2f}x")

        if scaling_factor > 1.1:
            print(f"✅ HiDPI scaling properly detected")

        return True

    except Exception as e:
        print(f"❌ Capture test failed: {e}")
        return False


def check_vlm_client() -> bool:
    """
    Test VLM client initialization and basic functionality.

    Returns:
        True if VLM client works correctly, False otherwise
    """
    print("\n=== VLM CLIENT TEST ===")

    try:
        # Test Gemini client initialization
        print("Testing Gemini client initialization...")
        try:
            gemini_client = VLMClient("gemini", "gemini-2.5-flash")
            print("✅ Gemini client initialized successfully")
            gemini_works = True
        except Exception as e:
            print(f"❌ Gemini client failed: {e}")
            gemini_works = False

        # Test OpenAI client initialization
        print("\nTesting OpenAI client initialization...")
        try:
            openai_client = VLMClient("openai", "openai/gpt-4.1")
            print("✅ OpenAI client initialized successfully")
            openai_works = True
        except Exception as e:
            print(f"❌ OpenAI client failed: {e}")
            openai_works = False

        # Test image preparation
        print("\nTesting image preparation for VLM...")
        try:
            test_image = capture_screen(monitor_index=1)
            encoded = prepare_for_vlm(test_image)

            # Check encoding size
            encoded_size_mb = len(encoded) / (1024 * 1024)
            print(f"  Encoded image size: {encoded_size_mb:.2f} MB")

            if encoded_size_mb > 10:
                print("⚠️  Warning: Large image size may cause API issues")
            else:
                print("✅ Image size is reasonable for API transmission")

        except Exception as e:
            print(f"❌ Image preparation failed: {e}")
            return False

        return gemini_works or openai_works

    except Exception as e:
        print(f"❌ VLM client test failed: {e}")
        return False


def check_projector() -> bool:
    """
    Test TelloActionProjector functionality with current monitor configuration.

    Returns:
        True if projector works correctly, False otherwise
    """
    print("\n=== ACTION PROJECTOR CHECK ===")

    # Create output directory
    output_path = Path("output/diagnostics")
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Get monitor info for proper configuration
        monitor_info = get_monitor_info()
        if not monitor_info:
            print("❌ No monitor info available for projector test")
            return False

        # Capture test image
        test_image = capture_screen(monitor_index=1)
        image_height, image_width = test_image.shape[:2]

        # Create TelloActionProjector with captured image dimensions
        try:
            action_projector = TelloActionProjector(
                image_width=image_width,
                image_height=image_height,
                mode="adaptive_mode"
            )
            print(f"✅ TelloActionProjector initialized ({image_width}x{image_height})")
        except Exception as e:
            print(f"❌ TelloActionProjector initialization failed: {e}")
            return False

        # Test point projection
        print("\nTesting 3D to 2D point projection...")
        test_points = [
            (0, 1, 0),    # Center, forward
            (1, 1, 0),    # Right, forward
            (-1, 1, 0),   # Left, forward
            (0, 1, 1),    # Center, forward, up
            (0, 1, -1)    # Center, forward, down
        ]

        center_point = action_projector.project_point((0, 1, 0))
        expected_center = (action_projector.image_width // 2, action_projector.image_height // 2)

        print(f"  Center point (0,1,0) projects to: {center_point}")
        print(f"  Expected center: {expected_center}")

        # Check projection accuracy
        x_diff = abs(center_point[0] - expected_center[0])
        y_diff = abs(center_point[1] - expected_center[1])

        if x_diff <= 10 and y_diff <= 10:
            print("✅ Point projection appears accurate")
            projection_accurate = True
        else:
            print(f"❌ Point projection may be inaccurate (off by {x_diff}, {y_diff} pixels)")
            projection_accurate = False

        # Create visualization
        viz_image = test_image.copy()

        for i, point_3d in enumerate(test_points):
            try:
                screen_point = action_projector.project_point(point_3d)

                # Draw point
                cv2.circle(viz_image, screen_point, 8, (0, 255, 0), -1)
                cv2.putText(
                    viz_image,
                    f"P{i+1}: {point_3d}",
                    (screen_point[0] + 10, screen_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

            except Exception as e:
                print(f"  Warning: Could not project point {point_3d}: {e}")

        # Save visualization
        cv2.imwrite(
            str(output_path / "projector_test.jpg"),
            cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
        )
        print(f"  Visualization saved to {output_path / 'projector_test.jpg'}")

        return projection_accurate

    except Exception as e:
        print(f"❌ TelloActionProjector test failed: {e}")
        return False


def check_encoding() -> bool:
    """
    Test image encoding pipeline for VLM APIs.

    Returns:
        True if encoding works correctly, False otherwise
    """
    print("\n=== IMAGE ENCODING TEST ===")

    # Create output directory
    output_path = Path("output/diagnostics")
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Capture test image
        image = capture_screen(monitor_index=1)
        print(f"Original image dimensions: {image.shape[1]}x{image.shape[0]}")

        # Save original
        cv2.imwrite(
            str(output_path / "encoding_original.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        # Test encoding
        encoded = prepare_for_vlm(image)
        print(f"Base64 encoded length: {len(encoded)} characters")

        # Test decoding to verify integrity
        decoded_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(decoded_bytes, np.uint8)
        decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if decoded_img is None:
            print("❌ Failed to decode image")
            return False

        print(f"Decoded image dimensions: {decoded_img.shape[1]}x{decoded_img.shape[0]}")

        # Save decoded image
        cv2.imwrite(str(output_path / "encoding_decoded.jpg"), decoded_img)

        # Check if dimensions match
        if decoded_img.shape[:2] == (image.shape[0], image.shape[1]):
            print("✅ Encoded/decoded dimensions match original")
            return True
        else:
            print("❌ Encoded/decoded dimensions do not match original")
            return False

    except Exception as e:
        print(f"❌ Encoding test failed: {e}")
        return False


def run_all_checks() -> bool:
    """
    Run all diagnostic checks and return overall system health.

    Returns:
        True if all critical checks pass, False otherwise
    """
    print("=== SPF SYSTEM DIAGNOSTICS ===")
    print("Running comprehensive system checks...")

    results = {}

    # Monitor check
    try:
        results["monitors"] = check_monitors()
    except Exception as e:
        print(f"❌ Monitor check failed: {e}")
        results["monitors"] = False

    # Capture check
    try:
        results["capture"] = check_capture()
    except Exception as e:
        print(f"❌ Capture check failed: {e}")
        results["capture"] = False

    # VLM client check
    try:
        results["vlm_client"] = check_vlm_client()
    except Exception as e:
        print(f"❌ VLM client check failed: {e}")
        results["vlm_client"] = False

    # Encoding check
    try:
        results["encoding"] = check_encoding()
    except Exception as e:
        print(f"❌ Encoding check failed: {e}")
        results["encoding"] = False

    # Projector check
    try:
        results["projector"] = check_projector()
    except Exception as e:
        print(f"❌ Projector check failed: {e}")
        results["projector"] = False

    # Generate summary
    print("\n=== DIAGNOSTIC SUMMARY ===")
    passed_count = 0
    total_count = len(results)

    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check.upper().replace('_', ' ')}: {status}")
        if passed:
            passed_count += 1

    # Overall result
    success_rate = (passed_count / total_count) * 100
    print(f"\nOverall: {passed_count}/{total_count} checks passed ({success_rate:.1f}%)")

    if success_rate >= 80:
        print("✅ System is healthy and ready for operation")
        return True
    elif success_rate >= 60:
        print("⚠️  System has some issues but may still function")
        return True
    else:
        print("❌ System has critical issues that need attention")
        return False


def save_diagnostic_report(results: Dict[str, bool], output_path: Path) -> None:
    """
    Save diagnostic results to a JSON report file.

    Args:
        results: Dictionary of check results
        output_path: Path to save the report
    """
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "platform": sys.platform,
            "python_version": sys.version,
        },
        "checks": results,
        "summary": {
            "total_checks": len(results),
            "passed_checks": sum(results.values()),
            "success_rate": (sum(results.values()) / len(results)) * 100
        }
    }

    with open(output_path / "diagnostic_report.json", 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    # Run all checks when script is executed directly
    success = run_all_checks()
    sys.exit(0 if success else 1)
