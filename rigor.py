#!/usr/bin/env python3
"""Thin shim so `python3 rigor.py ...` still works from a plain checkout,
without installing the package. See rigor/cli.py for the actual CLI; if
rigor is installed (`pip install rigor-mcp`), prefer the `rigor` command
it provides instead of this file.
"""
import sys

from rigor.cli import main

if __name__ == "__main__":
    sys.exit(main())
