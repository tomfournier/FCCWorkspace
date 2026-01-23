"""FCC ZH analysis package with optimized lazy imports.

This package provides analysis tools organized into submodules:
- config: Core configuration, constants, and utilities
- userConfig: User-facing configuration and path management
- plots: Plotting and visualization utilities
- func: BDT training and pseudo-data generation functions
- tools: Utility functions and process management

All heavy dependencies (numpy, pandas, xgboost, ROOT) are lazy-loaded
to minimize import time when only lightweight modules are needed.
"""

# Package metadata
__version__ = '1.0.0'
__author__ = 'FCC Analysis Team'

# Lazy imports will be provided by submodules on demand
# Users should import specific submodules or functions rather than using this __init__
