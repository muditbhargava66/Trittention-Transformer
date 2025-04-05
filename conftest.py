"""
Configuration file for pytest.

This file contains configuration settings and fixtures for pytest.
"""

import pytest
import warnings

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message="The configuration option \"asyncio_default_fixture_loop_scope\" is unset",
    category=DeprecationWarning
)

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=DeprecationWarning
)

warnings.filterwarnings(
    "ignore",
    message="declare_namespace",
    category=DeprecationWarning
)

# Configure pytest
def pytest_configure(config):
    """Configure pytest."""
    # Set asyncio mode and loop scope
    config.option.asyncio_mode = "strict"
    
    # Set asyncio fixture loop scope to avoid warning
    if not hasattr(config.option, "asyncio_fixture_loop_scope"):
        config.option.asyncio_fixture_loop_scope = "session"
