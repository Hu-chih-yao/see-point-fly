# Gemini API Testing Tools

This directory contains tools for testing and optimizing the Gemini API integration within the drone navigation project.

## Contents

- **accuracy_tester.py** - Core implementation for evaluating Gemini's accuracy
- **run_accuracy_test.py** - Wrapper script with proper path handling for easy execution

## Accuracy Tester

The accuracy testing tools provide a comprehensive framework for testing how well the Gemini model performs at generating navigation points from drone camera images.

### Features

- Test two prompt variations (baseline and precise) to identify optimal prompting strategy
- Run tests on static images with user-specified navigation instructions
- Generate visual comparisons of results between different prompt types
- Calculate performance metrics like processing time and spatial accuracy
- Save detailed test results for further analysis

### Retina Display Support

This tool properly handles macOS Retina/HiDPI displays which use a 2x scaling factor:

- ActionProjector is configured with the actual image dimensions (3420×2214) rather than the reported dimensions (1710×1107)
- Coordinates are properly converted between normalized (0-1000 scale) and actual pixel coordinates
- Visualizations are scaled appropriately for high-resolution displays
- Debug information shows both pixel coordinates and normalized coordinates

For more details on the resolution scaling issue, see `documents/monitor_resolution_findings.md`.

### Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- The Gemini API key must be set in your environment variables

### Usage

```bash
# Run from the project root directory
python -m tools.gemini.run_accuracy_test --tasks "fly toward the car" "navigate to the building"

# Or use the wrapper script directly
cd tools/gemini
./run_accuracy_test.py --tasks "fly toward the car" "navigate to the building"
```

### Required Arguments

- `--tasks`: List of navigation tasks to test (in quotes)

### Input/Output Paths

- Input images are automatically loaded from `tools/gemini/test_images/`
- Results are saved to `output/gemini_tests/[timestamp]/`

### Prompt Variations

The tool tests two prompt variations to identify which yields the most accurate results:

1. **Baseline**: Uses the default prompt from ActionProjector
2. **Precise**: Balanced prompt with spatial constraints and clear formatting requirements

### Output

The tool generates several outputs:

1. **Visualization grids**: Side-by-side comparisons of how each prompt performed on the same task
2. **Individual visualizations**: Detailed visualization of each test case
3. **JSON results**: Complete test data for programmatic analysis
4. **Performance metrics**: Statistical analysis of prompt performance
5. **Metric visualizations**: Charts comparing processing time and spatial distribution

### Interpreting Results

When analyzing results, look for:

1. **Consistency**: Which prompt variation produces the most consistent results?
2. **Accuracy**: Which points align best with the navigation task?
3. **Processing time**: Which prompts are most efficient?
4. **Point distribution**: Are the generated points in reasonable locations?

### Troubleshooting

- **Missing images**: Ensure test images are placed in the `test_images` directory
- **API errors**: Verify that your Gemini API key is set correctly
- **Task errors**: Make sure to provide at least one task with the `--tasks` argument
- **Display scaling**: The tool is configured for 2x Retina displays - if using a different display, update the parameters in `accuracy_tester.py`

## License

This tool is part of the drone navigation project and follows the same licensing terms. 