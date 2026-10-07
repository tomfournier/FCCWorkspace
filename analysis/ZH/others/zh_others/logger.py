"""
Centralized logging configuration for FCC analysis scripts.

This module provides a unified logging system for all analysis and package modules.
It ensures consistent log formatting and verbosity across the entire analysis pipeline.

Usage in analysis scripts:
    from package.logger import setup_logging, get_logger

    # In your main script (e.g., 2-BDT/train_bdt.py)
    setup_logging(verbose=args.verbose)
    LOGGER = get_logger(__name__)

    # Now use LOGGER throughout your script
    LOGGER.debug('This only appears with -v flag')
    LOGGER.info('This always appears')
    LOGGER.warning('Warning message')
    LOGGER.error('Error message')

Usage in package modules:
    from package.logger import get_logger

    # In your module (e.g., package/func/bdt.py)
    LOGGER = get_logger(__name__)

    # Use LOGGER just like in analysis scripts
    LOGGER.debug('Debugging this function')
    LOGGER.info('Processing complete')
"""

import logging
import sys


# Custom formatter similar to FCCAnalyses
class MultiLineFormatter(logging.Formatter):
    """
    Format log messages with proper indentation for multi-line messages.

    This ensures that when a log message spans multiple lines, all continuation
    lines are indented to align with the first line for better readability.
    """

    def get_header_length(self, record):
        """Calculate the length of the log message header."""
        return len(super().format(logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg='', args=(), exc_info=None
        )))

    def format(self, record):
        """Format record with indentation for continuation lines."""
        indent = ' ' * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(True)
        return head + ''.join(indent + line for line in trailing)


# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(verbose: bool = False, logger_name: str = 'FCCAnalysis') -> None:
    """
    Configure the root logging system for all analysis scripts and modules.

    This function should be called ONCE at the start of your main analysis script.
    All modules will then inherit this configuration when they call get_logger().

    Parameters
    ----------
    verbose : bool, optional
        If True, set logging level to DEBUG (show all messages including debug).
        If False, set to INFO (show only info/warning/error). Default: False
    logger_name : str, optional
        Name of the root logger. Default: 'FCCAnalysis'

    Notes
    -----
    - Call this function only once at the start of your main script
    - Package modules do NOT need to call this; they just call get_logger()
    - This mimics the setup in FCCAnalyses/bin/fccanalysis

    Examples
    --------
    In your main analysis script (e.g., 2-BDT/train_bdt.py):

        from package.logger import setup_logging, get_logger
        from package.parsing import create_parser, parse_args

        parser = create_parser(cat_single=True, include_sels=True)
        args = parse_args(parser)

        # Configure logging FIRST, before any other imports
        setup_logging(verbose=args.verbose)

        # Now get your logger
        LOGGER = get_logger(__name__)
        LOGGER.debug('Debugging info')
        LOGGER.info('Running analysis')
    """
    global _logging_configured

    if _logging_configured:
        # Prevent reconfiguration (logging should be set up only once)
        return

    # Get the root logger with our chosen name
    root_logger = logging.getLogger(logger_name)

    # Set the logging level based on verbose flag
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root_logger.setLevel(level)

    # Create formatter with the custom multi-line formatter
    formatter = MultiLineFormatter(
        fmt='[%(levelname)s]: %(message)s'
    )

    # Create and configure stream handler (console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(stream_handler)

    # Mark as configured
    _logging_configured = True

    # Log the setup
    root_logger.debug(f'Logging configured with level: {logging.getLevelName(level)}')


def get_logger(name: str, logger_root: str = 'FCCAnalysis') -> logging.Logger:
    """
    Get a logger instance for a module or script.

    This should be called in every module that needs logging. It returns a logger
    with a hierarchical name, allowing for organized log output. The logger inherits
    the configuration from setup_logging().

    Parameters
    ----------
    name : str
        The name of the logger, typically __name__ of your module.
        Examples: 'package.func.bdt', '__main__'
    logger_root : str, optional
        The root logger name. Must match what was passed to setup_logging().
        Default: 'FCCAnalysis'

    Returns
    -------
    logging.Logger
        A logger instance configured according to the root logger settings.

    Notes
    -----
    - Always use __name__ for the name parameter for proper hierarchy
    - The logger inherits settings from the root logger
    - Multiple calls with the same name return the same logger instance

    Examples
    --------
    In your main analysis script:

        from package.logger import get_logger

        LOGGER = get_logger(__name__)
        LOGGER.info('This is the main script')

    In a package module (e.g., package/func/bdt.py):

        from package.logger import get_logger

        LOGGER = get_logger(__name__)

        def train_model(data):
            LOGGER.debug('Starting model training')
            # ... training code ...
            LOGGER.info('Model training complete')
    """
    # Handle __main__ case - convert to a meaningful name
    if name == '__main__':
        logger_name = logger_root
    else:
        # Create hierarchical logger name: FCCAnalysis.package.func.bdt
        logger_name = f'{logger_root}.{name}'

    return logging.getLogger(logger_name)


def get_verbosity_level(logger_name: str = 'FCCAnalysis') -> str:
    """
    Get the current verbosity level as a string.

    Useful for logging what level is currently active.

    Parameters
    ----------
    logger_name : str, optional
        The root logger name. Default: 'FCCAnalysis'

    Returns
    -------
    str
        The logging level name ('DEBUG', 'INFO', 'WARNING', 'ERROR', etc.)
    """
    logger = logging.getLogger(logger_name)
    return logging.getLevelName(logger.level)
