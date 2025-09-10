"""
VLM (Vision Language Model) Tools for SPF Framework

This module provides tools for testing and evaluating VLM performance in navigation
tasks. It includes accuracy testing, prompt optimization, and performance analysis
capabilities that work with the unified VLMClient interface.

Classes:
    VLMAccuracyTester: Comprehensive accuracy testing for VLM navigation points

Functions:
    run_accuracy_test: Quick function to run accuracy tests with default settings
"""

from .accuracy_tester import VLMAccuracyTester

def run_accuracy_test(instructions, providers=None, output_dir=None):
    """
    Quick function to run VLM accuracy tests with default settings.

    Args:
        instructions: List of navigation instructions to test
        providers: List of (provider, model) tuples (defaults to gemini and openai)
        output_dir: Directory to save results (defaults to output/vlm_tests)

    Returns:
        Test results dictionary
    """
    if providers is None:
        providers = [
            ("gemini", "gemini-2.0-flash"),
            ("openai", "openai/gpt-4.1")
        ]

    tester = VLMAccuracyTester(output_dir=output_dir)
    results = tester.run_accuracy_tests(providers, instructions)
    analysis = tester.analyze_results(results)

    return {
        "results": results,
        "analysis": analysis,
        "output_dir": tester.results_dir
    }

__all__ = [
    "VLMAccuracyTester",
    "run_accuracy_test"
]
