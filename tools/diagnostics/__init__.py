"""
Diagnostic tools for checking monitor configuration, capture, and processing pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root) 