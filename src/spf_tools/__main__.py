#!/usr/bin/env python3
"""
SPF Tools Module Entry Point

This allows the spf_tools package to be executed as a module:
    python -m spf_tools [command] [options]

It provides a convenient command-line interface to all SPF tools functionality.
"""

from .cli import main

if __name__ == "__main__":
    exit(main())
