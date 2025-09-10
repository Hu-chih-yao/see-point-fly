# SPF Tools - Diagnostic and Utility Tools

This package provides comprehensive diagnostic and utility tools for the SPF (See, Point, Fly) framework. The tools are organized into specialized modules for different aspects of system management and testing.

## Overview

SPF Tools helps ensure your drone navigation system is properly configured and functioning optimally. It includes tools for monitor resolution management, screen capture testing, VLM accuracy evaluation, and comprehensive system diagnostics.

## Command Line Interface

SPF Tools can be used via the command line interface:

```bash
spf-tools diagnostics           # Run all system checks
spf-tools capture --test        # Test screen capture
spf-tools vlm --instructions "fly to car" --quick
spf-tools resolution --check    # Check monitor configuration
spf-tools monitors              # Show monitor summary
```

## Installation

SPF Tools is included as part of the SPF framework. No separate installation is required.

```python
from spf_tools import run_all_checks, capture_screen
from spf_tools.vlm import VLMAccuracyTester
from spf_tools.diagnostics import check_monitors
```

## Quick Start

### Run Complete System Check

```python
from spf_tools import run_all_checks

# Run all diagnostic checks
success = run_all_checks()
if success:
    print("System is ready for operation!")
```

### Test Screen Capture

```python
from spf_tools.capture import capture_screen, get_monitor_info

# Get monitor information
monitors = get_monitor_info()
print(f"Detected {len(monitors)} monitors")

# Capture screen
image = capture_screen(monitor_index=1)
print(f"Captured image: {image.shape[1]}x{image.shape[0]}")
```

### Test VLM Accuracy

```python
from spf_tools.vlm import run_accuracy_test

# Quick accuracy test
results = run_accuracy_test(
    instructions=["fly toward the car", "navigate to the building"],
    providers=[("gemini", "gemini-2.0-flash"), ("openai", "openai/gpt-4.1")]
)
```

## Module Documentation

### 📊 Diagnostics (`spf_tools.diagnostics`)

Comprehensive system health checks and monitoring tools.

**Key Functions:**
- `run_all_checks()`: Execute complete system diagnostic
- `check_monitors()`: Verify monitor configuration
- `check_capture()`: Test screen capture functionality
- `check_vlm_client()`: Test VLM client initialization
- `check_projector()`: Validate ActionProjector configuration

**Usage:**
```python
from spf_tools.diagnostics import check_monitors, check_vlm_client

# Check specific components
monitor_ok = check_monitors()
vlm_ok = check_vlm_client()
```

### 📷 Capture (`spf_tools.capture`)

Screen capture utilities with HiDPI/Retina display support.

**Key Functions:**
- `capture_screen()`: Capture with automatic scaling detection
- `capture_screen_resized()`: Capture and resize to logical resolution
- `prepare_for_vlm()`: Prepare images for VLM API transmission
- `get_monitor_info()`: Get detailed monitor configuration

**Usage:**
```python
from spf_tools.capture import capture_screen, prepare_for_vlm

# Capture and prepare for VLM
image = capture_screen(monitor_index=1)
encoded = prepare_for_vlm(image)
```

### 🧠 VLM Tools (`spf_tools.vlm`)

VLM testing and accuracy evaluation tools supporting multiple providers.

**Key Classes:**
- `VLMAccuracyTester`: Comprehensive accuracy testing framework

**Key Functions:**
- `run_accuracy_test()`: Quick accuracy testing with default settings

**Usage:**
```python
from spf_tools.vlm import VLMAccuracyTester

# Detailed accuracy testing
tester = VLMAccuracyTester(
    output_dir="my_tests",
    image_width=1920,
    image_height=1080
)

results = tester.run_accuracy_tests(
    providers=[("gemini", "gemini-2.0-flash"), ("openai", "openai/gpt-4.1")],
    instructions=["fly to the red car", "navigate around the building"]
)

analysis = tester.analyze_results(results)
```

### 🖥️ Resolution (`spf_tools.resolution`)

Monitor resolution management and scaling detection tools.

**Key Functions:**
- `check_monitors()`: Check monitor configurations
- `detect_scaling()`: Detect display scaling factors
- `generate_capture_fix()`: Generate resolution-corrected capture functions
- `validate_projector_config()`: Validate ActionProjector configuration

**Usage:**
```python
from spf_tools.resolution import check_monitors, validate_projector_config

# Check monitor setup
monitors = check_monitors()

# Validate projector configuration
validation = validate_projector_config(1920, 1080)
print(f"Config status: {validation['recommended_action']}")
```

## Common Use Cases

### 1. Initial System Setup

When setting up SPF on a new system:

```python
from spf_tools import run_all_checks
from spf_tools.resolution import print_monitor_summary

# Check system configuration
print_monitor_summary()
system_ok = run_all_checks()

if not system_ok:
    print("Please address the issues above before proceeding")
```

### 2. HiDPI Display Configuration

For systems with HiDPI/Retina displays:

```python
from spf_tools.resolution import detect_scaling, generate_capture_fix

# Detect scaling
width_scale, height_scale = detect_scaling(1)
print(f"Detected scaling: {width_scale:.2f}x")

if width_scale > 1.1:
    # Generate resolution fix
    generate_capture_fix("resolution_fix.py")
    print("Resolution fix generated - update your capture code")
```

### 3. VLM Performance Optimization

To find the best VLM provider and prompt configuration:

```python
from spf_tools.vlm import VLMAccuracyTester

tester = VLMAccuracyTester()
results = tester.run_accuracy_tests(
    providers=[
        ("gemini", "gemini-2.0-flash"),
        ("gemini", "gemini-2.5-pro"),
        ("openai", "openai/gpt-4.1")
    ],
    instructions=[
        "fly toward the red vehicle",
        "navigate to the tall building",
        "approach the landing pad"
    ]
)

# Results include accuracy metrics, processing times, and visualizations
analysis = tester.analyze_results(results)
```

### 4. Troubleshooting Capture Issues

When screen capture produces unexpected results:

```python
from spf_tools.capture import get_monitor_info, capture_screen
from spf_tools.diagnostics import check_capture

# Get detailed monitor information
monitors = get_monitor_info()
for name, info in monitors.items():
    print(f"{name}: {info['logical_width']}x{info['logical_height']} "
          f"(scaling: {info['width_scaling']:.2f}x)")

# Test capture functionality
capture_ok = check_capture()
```

### 5. ActionProjector Configuration Validation

To ensure ActionProjector dimensions match your display:

```python
from spf_tools.resolution import validate_projector_config
from spf.projectors.action_projector import ActionProjector

# Get current ActionProjector configuration
projector = ActionProjector()
validation = validate_projector_config(
    projector.image_width,
    projector.image_height
)

print(f"Recommendation: {validation['recommended_action']}")
```

## Output Files

SPF Tools generates various output files for analysis and debugging:

### Diagnostic Reports
- `output/diagnostics/diagnostic_report.json`: Complete system diagnostic data
- `output/diagnostics/capture_*.jpg`: Screen capture test images
- `output/diagnostics/projector_test.jpg`: ActionProjector visualization

### VLM Test Results
- `output/vlm_tests/[timestamp]/test_results.json`: Complete test data
- `output/vlm_tests/[timestamp]/analysis.json`: Performance analysis
- `output/vlm_tests/[timestamp]/*_comparison.png`: Visual comparisons
- `output/vlm_tests/[timestamp]/metrics_comparison.png`: Metric charts

### Resolution Fixes
- `capture_screen_fixed.py`: Auto-generated capture fix for HiDPI displays

## Configuration

### Environment Variables

SPF Tools respects the same environment variables as the main SPF framework:

- `GEMINI_API_KEY`: Required for Gemini VLM testing
- `OPENAI_API_KEY`: Required for OpenAI VLM testing
- `OPENAI_BASE_URL`: Custom OpenAI API endpoint (defaults to OpenRouter)

### Test Images

For VLM accuracy testing, place test images in:
- `src/spf_tools/vlm/test_images/` (default)
- Or specify custom directory with `test_images_dir` parameter

Supported formats: `.jpg`, `.jpeg`, `.png`

## Troubleshooting

### Common Issues

**Monitor scaling problems:**
```python
# Check if scaling is detected
from spf_tools.resolution import detect_scaling
scaling = detect_scaling(1)
if scaling[0] > 1.1:
    print("HiDPI display detected - use capture_screen_resized() or generate fix")
```

**VLM API errors:**
```python
# Test VLM client initialization
from spf_tools.diagnostics import check_vlm_client
if not check_vlm_client():
    print("Check your API keys and network connection")
```

**Dimension mismatches:**
```python
# Validate ActionProjector configuration
from spf_tools.resolution import validate_projector_config
validation = validate_projector_config(1920, 1080)
print(f"Action needed: {validation['recommended_action']}")
```

### Getting Help

1. Run complete diagnostics: `from spf_tools import run_all_checks; run_all_checks()`
2. Check the generated diagnostic reports in `output/diagnostics/`
3. Review monitor configuration with `print_monitor_summary()`
4. Test individual components with specific diagnostic functions

## API Reference

For detailed API documentation, see the individual module documentation:

- [Diagnostics API](diagnostics/__init__.py)
- [Capture API](capture/__init__.py)
- [VLM Tools API](vlm/__init__.py)
- [Resolution API](resolution/__init__.py)

## Examples

## Command Line Usage

The `spf-tools` command provides convenient access to all functionality:

```bash
# System diagnostics
spf-tools diagnostics --all
spf-tools diagnostics --monitors --capture

# Screen capture testing
spf-tools capture --info --test --save

# VLM accuracy testing
spf-tools vlm --instructions "navigate to building" "fly to car" --providers gemini:gemini-2.0-flash openai:openai/gpt-4.1

# Resolution management
spf-tools resolution --check --fix
spf-tools monitors
```

---

For detailed command-line options, use `spf-tools [command] --help`.
