# VLM Test Images

This directory contains test images for VLM (Vision Language Model) accuracy testing. Place your test images here to evaluate VLM performance with different navigation instructions.

## Supported Formats

- `.jpg` / `.jpeg`
- `.png`

## Image Requirements

- Images should represent typical drone camera views
- Recommended resolution: 1920x1080 or higher
- Clear visibility of navigation targets (vehicles, buildings, landmarks)
- Varied lighting conditions and environments for comprehensive testing

## Usage

The VLM accuracy tester will automatically discover and use all supported images in this directory:

```python
from spf_tools.vlm import VLMAccuracyTester

tester = VLMAccuracyTester()
results = tester.run_accuracy_tests(
    providers=[("gemini", "gemini-2.5-flash")],
    instructions=["fly toward the car", "navigate to the building"]
)
```

Or via command line:

```bash
spf-tools vlm --instructions "fly to car" "navigate to building"
```

## Test Image Guidelines

**Good test images include:**
- Clear target objects that match common navigation instructions
- Multiple objects of the same type for selection testing
- Various distances and angles
- Different environmental conditions (urban, rural, indoor, outdoor)

**Avoid:**
- Extremely blurry or low-contrast images
- Images with no clear navigation targets
- Copyrighted content

## Example Instructions to Test

- "fly toward the red car"
- "navigate to the tall building"
- "approach the landing pad"
- "move toward the person"
- "go to the tree"
- "fly to the parking lot"
- "navigate around the obstacle"

Place your test images in this directory and run VLM accuracy tests to evaluate performance across different providers and prompt configurations.
