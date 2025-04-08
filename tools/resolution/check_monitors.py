#!/usr/bin/env python3
"""
Monitor resolution checker
Checks and reports all monitor resolutions
"""

import sys
import os
import mss
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from drone_controller import print_monitor_info
except ImportError:
    # Define backup function in case import fails
    def print_monitor_info(monitor_id=None):
        print(f"Using backup print_monitor_info function")
        with mss.mss() as sct:
            if monitor_id is not None and monitor_id < len(sct.monitors):
                mon = sct.monitors[monitor_id]
                print(f"Monitor {monitor_id}: {mon}")
                print(f"  Dimensions: {mon['width']}x{mon['height']}")
                print(f"  Position: left={mon['left']}, top={mon['top']}")
            else:
                for idx, mon in enumerate(sct.monitors):
                    print(f"Monitor {idx}: {mon}")
                    print(f"  Dimensions: {mon['width']}x{mon['height']}")
                    print(f"  Position: left={mon['left']}, top={mon['top']}")

def check_monitor_resolutions():
    """Check and report all monitor resolutions"""
    print("\n=== MONITOR RESOLUTIONS ===")
    
    try:
        with mss.mss() as sct:
            print(f"Total monitors: {len(sct.monitors)}")
            
            # Monitor 0 is special - it's the "all monitors" virtual screen
            print(f"\nMonitor 0 (All monitors combined):")
            m0 = sct.monitors[0]
            print(f"  Dimensions: {m0['width']}x{m0['height']}")
            print(f"  Position: Left={m0['left']}, Top={m0['top']}")
            
            # Print individual monitor details
            for i in range(1, len(sct.monitors)):
                monitor = sct.monitors[i]
                print(f"\nMonitor {i}:")
                print(f"  Dimensions: {monitor['width']}x{monitor['height']}")
                print(f"  Position: Left={monitor['left']}, Top={monitor['top']}")
    except Exception as e:
        print(f"Error checking monitors: {e}")

if __name__ == "__main__":
    print("=== QUICK MONITOR RESOLUTION CHECK ===")
    check_monitor_resolutions()
    print("\nCheck complete.") 