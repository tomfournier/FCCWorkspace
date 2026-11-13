"""
Centralized logging configuration for FCC analysis scripts.

This module provides a unified logging system for all analysis and package modules.
It ensures consistent log formatting and verbosity across the entire analysis pipeline.

Usage in analysis scripts:

```python
from package.logger import setup_logging, get_logger

# In your main script (e.g., 2-BDT/train_bdt.py)
setup_logging(verbose=args.verbose)
LOGGER = get_logger(__name__)

# Now use LOGGER throughout your script
LOGGER.debug('This only appears with -v flag')
LOGGER.info('This always appears')
LOGGER.warning('Warning message')
LOGGER.error('Error message')
```

Usage in package modules:

```python
from package.logger import get_logger

# In your module (e.g., package/func/bdt.py)
LOGGER = get_logger(__name__)

# Use LOGGER just like in analysis scripts
LOGGER.debug('Debugging this function')
LOGGER.info('Processing complete')
```
"""

################################
### STANDARD LIBRARY IMPORTS ###
################################

import copy, os, sys, logging



####################################
### NEW LOGGING LEVEL DEFINITION ###
####################################

TRACE = 5
"""Logging level used for the most detailed diagnostic messages."""
logging.addLevelName(TRACE, 'DEBUG')

INSPECT = 25
"""Logging level for explicitly enabled variable inspection messages."""
logging.addLevelName(INSPECT, 'INSPECT')

INSP_DBG = 24
"""Logging level for more detailed inspection messages."""
logging.addLevelName(INSP_DBG, 'DEBUG')



#########################
### COLORS DEFINITION ###
#########################

ANSI_COLORS = {
    'black': 30, 'red': 31, 'green': 32, 'yellow': 33,
    'blue': 34, 'magenta': 35, 'cyan': 36, 'white': 37,
    'orange': '#FFA500',
}

def ansi_escape(*codes: int) -> str:
    """Build an ANSI escape sequence from numeric codes."""
    return '\033[' + ';'.join(str(code) for code in codes) + 'm'


def ansi_color(color: str | int, style: str | None = None,
               background: bool = False) -> tuple[str, str]:
    """Return color and reset codes for a name, #RRGGBB, or palette index.

    Style may be bold, thin, or light. Light uses the bright terminal palette
    for basic colors and blends RGB colors halfway toward white; it is not
    available for 256-color palette indices.
    """
    # Bold (1) and thin/dim (2) are attributes; light changes the color itself.
    style_codes = {None: (), 'bold': (1,), 'thin': (2,), 'light': ()}[style]

    # Extended foreground and background colors start with 38 and 48 respectively.
    channel = 48 if background else 38
    if isinstance(color, int):
        if not 0 <= color <= 255 or style == 'light':
            raise ValueError('Palette indices must be 0-255 and do not support light')
        # 38/48;5;N selects entry N in the terminal's 256-color palette.
        codes = (channel, 5, color)
    else:
        # Names resolve to basic ANSI codes or RGB values; hex strings pass through.
        color_code = ANSI_COLORS.get(color, color)
        if isinstance(color_code, int):
            # Background codes are 10 higher; bright variants are 60 higher.
            codes = (color_code + (10 if background else 0) + (60 if style == 'light' else 0),)
        else:
            if not color_code.startswith('#'):
                raise KeyError(color)
            try:
                # Convert #RRGGBB into three bytes, one per color channel.
                channels = bytes.fromhex(color_code[1:])
                if len(color_code) != 7 or len(channels) != 3:
                    raise ValueError
            except ValueError:
                raise ValueError(f'Invalid RGB color: {color_code!r}') from None
            if style == 'light':
                # Move each channel halfway toward white (255).
                channels = bytes((value + 255) // 2 for value in channels)
            # 38/48;2;R;G;B selects a 24-bit RGB color.
            codes = (channel, 2, *channels)
    # Return a separate reset so callers can limit styling to the intended text.
    return ansi_escape(*style_codes, *codes), ansi_escape(0)



#########################
### LOGGER DEFINITION ###
#########################

# Custom formatter similar to FCCAnalyses
class MultiLineFormatter(logging.Formatter):
    """
    Format log messages with proper indentation for multi-line messages.

    This ensures that when a log message spans multiple lines, all continuation
    lines are indented to align with the first line for better readability.
    """

    LEVEL_COLORS = {
        TRACE:            ('blue',   'bold'),
        logging.DEBUG:    ('blue',   None),
        logging.INFO:     ('white',  None),
        INSPECT:          ('green',  None),
        INSP_DBG:         ('green',  'bold'),
        logging.WARNING:  ('orange', None),
        logging.ERROR:    ('red',    None),
        logging.CRITICAL: ('red',    'bold'),
    }

    def __init__(self, *args, color=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color

    def get_header_length(self, record):
        """Calculate the length of the log message header."""
        return len(super().format(logging.LogRecord(
            record.name, record.levelno,
            record.pathname, record.lineno,
            '', (), None
        )))

    def format(self, record):
        """Format record with indentation for continuation lines."""
        indent = ' ' * self.get_header_length(record)
        color = self.LEVEL_COLORS.get(record.levelno) if self.color else None
        if color:
            color_code, reset = ansi_color(*color)
            record = copy.copy(record)
            record.levelname = f'{color_code}{record.levelname}{reset}'
        head, *trailing = super().format(record).splitlines(True)
        return head + ''.join(indent + line for line in trailing)


# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(verbose: int | bool = 0, logger_name: str = 'FCCAnalysis',
                  inspect: int | bool = 0) -> None:
    """
    Configure the root logging system for all analysis scripts and modules.

    This function should be called ONCE at the start of your main analysis script.
    All modules will then inherit this configuration when they call get_logger().

    Parameters
    ----------
    verbose : int or bool, optional
        Verbosity setting: 0 shows INFO and above, 1 enables DEBUG, 2 enables
        TRACE, and 3 or more enables all positive logging levels except the
        inspection levels unless requested separately. True is equivalent to 1.
        Default: 0
    logger_name : str, optional
        Name of the root logger. Default: 'FCCAnalysis'
    inspect : int or bool, optional
        0 disables inspection, 1 enables INSPECT, and 2 enables INSPECT_DEBUG
        as well. With verbosity 0, only inspection messages and WARNING or
        higher are shown. True is equivalent to 1. Default: 0

    Notes
    -----
    - Call this function only once at the start of your main script
    - Package modules do NOT need to call this; they just call get_logger()
    - This mimics the setup in FCCAnalyses/bin/fccanalysis
    - Level colors are enabled on terminals unless NO_COLOR is set

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

    # Prevent reconfiguration (logging should be set up only once)
    if _logging_configured: return

    # Get the root logger with our chosen name
    root_logger = logging.getLogger(logger_name)

    verbosity_levels = {0: logging.INFO, 1: logging.DEBUG, 2: TRACE}
    inspection_levels = {INSPECT: 1, INSP_DBG: 2}
    if inspect and int(verbose) == 0:
        level = INSP_DBG if inspect >= 2 else INSPECT
    else:
        level = verbosity_levels.get(int(verbose), 1)

    root_logger.setLevel(level)
    root_logger.propagate = False

    # Create formatter with the custom multi-line formatter
    formatter = MultiLineFormatter(fmt='[%(levelname)s]: %(message)s',
                                   color=sys.stdout.isatty() and 'NO_COLOR' not in os.environ)

    # Create and configure stream handler (console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.addFilter(lambda record: inspection_levels.get(record.levelno, 0) <= inspect)
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
    if name == '__main__': logger_name = logger_root
    # Create hierarchical logger name: FCCAnalysis.package.func.bdt
    else: logger_name = f'{logger_root}.{name}'

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
