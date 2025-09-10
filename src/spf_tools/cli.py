#!/usr/bin/env python3
"""
SPF Tools Command-Line Interface

This module provides a command-line interface for accessing SPF Tools functionality.
It allows users to run diagnostics, test VLM accuracy, check monitors, and perform
other maintenance tasks from the command line.

Usage:
    python -m spf_tools.cli [command] [options]

Commands:
    diagnostics     Run system diagnostic checks
    capture         Test screen capture functionality
    vlm             Run VLM accuracy tests
    resolution      Check monitor resolution and scaling
    monitors        Display monitor configuration
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Import SPF tools modules
try:
    from .diagnostics import (
        run_all_checks, check_monitors, check_capture,
        check_vlm_client, check_projector, check_encoding
    )
    from .capture import capture_screen, get_monitor_info, create_test_image
    from .vlm import VLMAccuracyTester, run_accuracy_test
    from .resolution import (
        check_monitors as check_monitor_config,
        detect_scaling, generate_capture_fix,
        validate_projector_config, print_monitor_summary
    )
except ImportError as e:
    print(f"Error importing SPF tools: {e}")
    print("Make sure SPF framework is properly installed")
    sys.exit(1)


def cmd_diagnostics(args) -> int:
    """Run system diagnostic checks."""
    print("=== SPF SYSTEM DIAGNOSTICS ===")

    if args.all or not any([args.monitors, args.capture, args.vlm, args.projector, args.encoding]):
        # Run all checks if no specific check is requested
        success = run_all_checks()
        return 0 if success else 1

    results = []

    if args.monitors:
        print("\nRunning monitor check...")
        results.append(check_monitors())

    if args.capture:
        print("\nRunning capture check...")
        results.append(check_capture())

    if args.vlm:
        print("\nRunning VLM client check...")
        results.append(check_vlm_client())

    if args.projector:
        print("\nRunning projector check...")
        results.append(check_projector())

    if args.encoding:
        print("\nRunning encoding check...")
        results.append(check_encoding())

    # Return success if all checks passed
    return 0 if all(results) else 1


def cmd_capture(args) -> int:
    """Test screen capture functionality."""
    print("=== SCREEN CAPTURE TEST ===")

    if args.info:
        # Display monitor information
        monitors = get_monitor_info()
        if monitors:
            for name, info in monitors.items():
                print(f"\n{name}:")
                print(f"  Logical: {info['width']}x{info['height']}")
                print(f"  Physical: {info['captured_width']}x{info['captured_height']}")
                print(f"  Scaling: {info['width_scaling']:.2f}x")
                print(f"  HiDPI: {'Yes' if info['is_hidpi'] else 'No'}")
        else:
            print("No monitors detected")
            return 1

    if args.test:
        # Test capture functionality
        try:
            print(f"\nCapturing from monitor {args.monitor}...")
            image = capture_screen(monitor_index=args.monitor)
            print(f"Captured image: {image.shape[1]}x{image.shape[0]} pixels")

            if args.save:
                import cv2
                output_path = Path("capture_test.jpg")
                cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print(f"Saved test capture to: {output_path}")

        except Exception as e:
            print(f"Capture test failed: {e}")
            return 1

    if args.create_test:
        # Create test image
        test_img = create_test_image(args.width or 1920, args.height or 1080)
        import cv2
        output_path = Path("test_pattern.jpg")
        cv2.imwrite(str(output_path), cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        print(f"Created test pattern: {output_path}")

    return 0


def cmd_vlm(args) -> int:
    """Run VLM accuracy tests."""
    print("=== VLM ACCURACY TESTING ===")

    if not args.instructions:
        print("Error: --instructions is required for VLM testing")
        return 1

    # Parse provider specifications
    providers = []
    for provider_spec in args.providers:
        if ":" in provider_spec:
            provider, model = provider_spec.split(":", 1)
        else:
            # Use default models
            if provider_spec == "gemini":
                provider, model = "gemini", "gemini-2.5-flash"
            elif provider_spec == "openai":
                provider, model = "openai", "openai/gpt-4.1"
            else:
                print(f"Warning: Unknown provider {provider_spec}, using default model")
                provider, model = provider_spec, "default"
        providers.append((provider, model))

    if args.quick:
        # Quick test using convenience function
        try:
            results = run_accuracy_test(
                instructions=args.instructions,
                providers=providers,
                output_dir=args.output
            )
            print(f"\n✅ Quick test complete!")
            print(f"Results saved to: {results['output_dir']}")
            return 0
        except Exception as e:
            print(f"Quick test failed: {e}")
            return 1
    else:
        # Detailed test using full tester
        try:
            tester = VLMAccuracyTester(
                output_dir=args.output,
                test_images_dir=args.images,
                image_width=args.width or 1920,
                image_height=args.height or 1080
            )

            results = tester.run_accuracy_tests(providers, args.instructions)
            analysis = tester.analyze_results(results)

            print(f"\n✅ Detailed test complete!")
            print(f"Results saved to: {tester.results_dir}")
            return 0
        except Exception as e:
            print(f"VLM test failed: {e}")
            return 1


def cmd_resolution(args) -> int:
    """Check monitor resolution and scaling."""
    print("=== RESOLUTION MANAGEMENT ===")

    if args.check:
        # Check monitor configuration
        monitors = check_monitor_config()
        if monitors:
            print(f"Detected {len(monitors)} monitor(s)")
            for name, info in monitors.items():
                print(f"\n{name}:")
                print(f"  Logical: {info['logical_width']}x{info['logical_height']}")
                print(f"  Physical: {info['physical_width']}x{info['physical_height']}")
                print(f"  Scaling: {info['width_scaling']:.2f}x")
                print(f"  HiDPI: {'Yes' if info['is_hidpi'] else 'No'}")
        else:
            print("No monitors detected")
            return 1

    if args.scaling:
        # Check scaling for specific monitor
        width_scale, height_scale = detect_scaling(args.monitor)
        print(f"\nMonitor {args.monitor} scaling:")
        print(f"  Width: {width_scale:.2f}x")
        print(f"  Height: {height_scale:.2f}x")

        if width_scale > 1.1 or height_scale > 1.1:
            print("  HiDPI display detected")
        else:
            print("  Standard display")

    if args.fix:
        # Generate resolution fix
        if generate_capture_fix(args.output):
            print(f"Resolution fix generated: {args.output or 'capture_screen_fixed.py'}")
        else:
            print("No fix needed - no scaling detected")

    if args.validate and args.width and args.height:
        # Validate projector configuration
        validation = validate_projector_config(args.width, args.height)
        print(f"\nProjector configuration validation:")
        print(f"  Matches logical dimensions: {'Yes' if validation['matches_logical'] else 'No'}")
        print(f"  Matches physical dimensions: {'Yes' if validation['matches_physical'] else 'No'}")
        print(f"  Scaling detected: {'Yes' if validation['scaling_detected'] else 'No'}")
        print(f"  Recommended action: {validation['recommended_action']}")

    return 0


def cmd_monitors(args) -> int:
    """Display monitor configuration summary."""
    print_monitor_summary()
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for SPF Tools CLI."""
    parser = argparse.ArgumentParser(
        description="SPF Tools - Diagnostic and utility tools for the SPF framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s diagnostics                    # Run all diagnostic checks
  %(prog)s diagnostics --monitors         # Check monitor configuration only
  %(prog)s capture --test --save          # Test capture and save result
  %(prog)s vlm --instructions "fly to car" --quick
  %(prog)s resolution --check --fix       # Check resolution and generate fix
  %(prog)s monitors                       # Show monitor summary
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Diagnostics command
    diag_parser = subparsers.add_parser('diagnostics', help='Run system diagnostic checks')
    diag_parser.add_argument('--all', action='store_true', help='Run all checks (default)')
    diag_parser.add_argument('--monitors', action='store_true', help='Check monitor configuration')
    diag_parser.add_argument('--capture', action='store_true', help='Test screen capture')
    diag_parser.add_argument('--vlm', action='store_true', help='Test VLM client')
    diag_parser.add_argument('--projector', action='store_true', help='Test ActionProjector')
    diag_parser.add_argument('--encoding', action='store_true', help='Test image encoding')

    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Test screen capture functionality')
    capture_parser.add_argument('--info', action='store_true', help='Show monitor information')
    capture_parser.add_argument('--test', action='store_true', help='Test capture functionality')
    capture_parser.add_argument('--save', action='store_true', help='Save captured image')
    capture_parser.add_argument('--create-test', action='store_true', help='Create test pattern image')
    capture_parser.add_argument('--monitor', type=int, default=1, help='Monitor index to test (default: 1)')
    capture_parser.add_argument('--width', type=int, help='Test image width')
    capture_parser.add_argument('--height', type=int, help='Test image height')

    # VLM command
    vlm_parser = subparsers.add_parser('vlm', help='Run VLM accuracy tests')
    vlm_parser.add_argument('--instructions', nargs='+', required=True,
                           help='Navigation instructions to test')
    vlm_parser.add_argument('--providers', nargs='+',
                           default=['gemini:gemini-2.5-flash', 'openai:openai/gpt-4.1'],
                           help='VLM providers to test (format: provider:model)')
    vlm_parser.add_argument('--images', help='Directory with test images')
    vlm_parser.add_argument('--output', help='Output directory for results')
    vlm_parser.add_argument('--width', type=int, help='Image width for coordinate conversion')
    vlm_parser.add_argument('--height', type=int, help='Image height for coordinate conversion')
    vlm_parser.add_argument('--quick', action='store_true', help='Run quick test with defaults')

    # Resolution command
    res_parser = subparsers.add_parser('resolution', help='Check monitor resolution and scaling')
    res_parser.add_argument('--check', action='store_true', help='Check monitor configuration')
    res_parser.add_argument('--scaling', action='store_true', help='Check scaling factors')
    res_parser.add_argument('--fix', action='store_true', help='Generate resolution fix')
    res_parser.add_argument('--validate', action='store_true', help='Validate projector config')
    res_parser.add_argument('--monitor', type=int, default=1, help='Monitor index (default: 1)')
    res_parser.add_argument('--width', type=int, help='Projector width for validation')
    res_parser.add_argument('--height', type=int, help='Projector height for validation')
    res_parser.add_argument('--output', help='Output file for generated fix')

    # Monitors command
    subparsers.add_parser('monitors', help='Display monitor configuration summary')

    return parser


def main() -> int:
    """Main entry point for SPF Tools CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    try:
        if args.command == 'diagnostics':
            return cmd_diagnostics(args)
        elif args.command == 'capture':
            return cmd_capture(args)
        elif args.command == 'vlm':
            return cmd_vlm(args)
        elif args.command == 'resolution':
            return cmd_resolution(args)
        elif args.command == 'monitors':
            return cmd_monitors(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
