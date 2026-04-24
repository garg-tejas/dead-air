"""Pytest configuration for Dead Air tests."""

import os
import sys

# Ensure repo root is on PYTHONPATH so `dead_air.server` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
