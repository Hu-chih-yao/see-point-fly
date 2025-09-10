#!/usr/bin/env python3
"""
SPF (See, Point, Fly) - Main Entry Point
A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation

This is the main entry point that allows users to choose between simulator
and real Tello drone modes via command line arguments.
"""

import sys
import argparse

# We're now inside the spf package, so imports work directly

def main():
    """Main entry point with mode selection"""
    parser = argparse.ArgumentParser(
        description='SPF (See, Point, Fly) - VLM Framework for Drone Navigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sim                    # Run in simulator mode
  %(prog)s tello                  # Run with real Tello drone
  %(prog)s sim --debug            # Run simulator with debug output
  %(prog)s tello --test           # Run Tello in test mode
  %(prog)s sim --info             # Show monitor information and exit
        """
    )

    parser.add_argument(
        'mode',
        choices=['sim', 'tello'],
        help='Operation mode: "sim" for simulator, "tello" for real drone'
    )

    # Common arguments
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with additional logging'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with static images'
    )

    # Simulator-specific arguments
    sim_group = parser.add_argument_group('simulator options')
    sim_group.add_argument(
        '--monitor',
        type=int,
        default=1,
        help='Monitor index for simulator (1=primary monitor)'
    )

    sim_group.add_argument(
        '--info',
        action='store_true',
        help='Display monitor information and exit (simulator only)'
    )

    # Tello-specific arguments
    tello_group = parser.add_argument_group('tello options')
    tello_group.add_argument(
        '--skip-camera-check',
        action='store_true',
        help='Skip camera initialization check (Tello only)'
    )

    tello_group.add_argument(
        '--record',
        action='store_true',
        help='Record frames continuously (Tello only)'
    )

    tello_group.add_argument(
        '--record-session',
        type=str,
        help='Name for the recording session (Tello only)'
    )

    args = parser.parse_args()

    # Import and run the appropriate mode
    if args.mode == 'sim':
        from .simulator_main import main as sim_main
        return sim_main(args)
    elif args.mode == 'tello':
        from .tello_main import main as tello_main
        return tello_main(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
