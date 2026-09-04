"""Test bootstrap: make `from modules import ...` work like it does inside aiva/."""
import os
import sys

AIVA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aiva")
if AIVA_DIR not in sys.path:
    sys.path.insert(0, AIVA_DIR)
