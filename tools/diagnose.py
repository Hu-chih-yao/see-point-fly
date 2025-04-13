#!/usr/bin/env python3
"""
Main diagnostic entry point for the drone navigation system.
Run this to check all components of the system.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_root)

from tools.diagnostics.system_check import (
    check_monitors,
    check_capture,
    check_encoding,
    check_projector,
    run_all_checks
)

def main():
    parser = argparse.ArgumentParser(description="Drone system diagnostics")
    parser.add_argument("--monitors", action="store_true", help="Check monitor configuration")
    parser.add_argument("--capture", action="store_true", help="Test screen capture")
    parser.add_argument("--encoding", action="store_true", help="Test image encoding")
    parser.add_argument("--projector", action="store_true", help="Test ActionProjector")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    
    args = parser.parse_args()
    
    # If no specific checks selected, run all
    if not any([args.monitors, args.capture, args.encoding, args.projector, args.all]):
        args.all = True
    
    if args.all:
        run_all_checks()
    else:
        if args.monitors:
            check_monitors()
        
        if args.capture:
            check_capture()
        
        if args.encoding:
            check_encoding()
        
        if args.projector:
            check_projector()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 