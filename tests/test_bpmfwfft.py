"""
Unit and regression test for the bpmfwfft package.
"""

# Import package, test suite, and other packages as needed
import bpmfwfft
import pytest
import sys

def test_bpmfwfft_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "bpmfwfft" in sys.modules
