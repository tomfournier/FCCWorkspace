'''
Centralized argument parsing for FCC analysis scripts.

This module provides flexible, modular argument parsing to avoid duplication
across analysis scripts. Arguments are organized into logical groups that can
be mixed and matched based on each script's needs.

Core principles:
- Minimal base functions with optional parameters
- Only add arguments that a script actually uses
- Easy to extend with new argument groups
- Reduced code duplication through composition

Examples:
    Script-specific parser:
        from package.parsing import create_parser
        parser = create_parser('2-BDT', 'evaluation')
        args = parser.parse_args()

'''

from argparse import ArgumentParser, Namespace, BooleanOptionalAction, _ArgumentGroup



# ========================================================== #
# MODULAR ARGUMENT BUILDERS (low-level, reusable components) #
# ========================================================== #

def add_cat_argument(
    parser: ArgumentParser,
    multi: bool = False,
    allow_empty: bool = False,
    default: str | None = None,
    allow_qq: bool = True,
    group: str | None = None
) -> None:
    '''Add --cat/--cats argument for final state selection.'''

    if default is None: def_value = 'ee-mumu' if multi else ''
    else:               def_value = default

    choices = ['ee', 'mumu', 'qq'] if allow_qq else ['ee', 'mumu']
    if multi:
        choices.extend(['ee-mumu', 'mumu-ee'])
        if allow_qq:
            choices.extend(['ee-qq', 'ee-mumu-qq', 'ee-qq-mumu',
                            'mumu-qq', 'mumu-ee-qq', 'mumu-qq-ee',
                            'qq-ee', 'qq-mumu', 'qq-ee-mumu', 'qq-mumu-ee'])
    if allow_empty: choices.append('')

    if group is None:
        group: _ArgumentGroup = parser.add_argument_group('General arguments')

    # Use metavar to show a concise pattern instead of listing all choices
    metavar = 'CHANNELS' if multi else 'CHANNEL'
    help_text = ('Final state lepton category: ee, mumu' + ', qq' if allow_qq else '' +
                 (' or combinations separated by dash like ee-mumu' if multi else '') +
                 f' (default: "{def_value}")')

    group.add_argument(
        '--cat', '--cats',
        type=str,
        default=def_value,
        choices=choices,
        metavar=metavar,
        help=help_text
    )


def add_ecm_argument(
    parser: ArgumentParser,
    multi: bool = False,
    default: int | str | None = None,
    group: str | None = None
) -> None:
    '''Add --ecm/--ecms argument for center-of-mass energy.'''
    if default is None:
        default = '240' if multi else 240
    else:
        if isinstance(default, int):
            default = str(default) if multi else default
        elif isinstance(default, str):
            default = int(default) if not multi else default
        else:
            raise TypeError('Either int or str type are supported')

    if group is None:
        group: _ArgumentGroup = parser.add_argument_group('General arguments')

    metavar = 'ENERGIES' if multi else 'ENERGY'
    help_text = ('Center-of-mass energy in GeV: 240 or 365' +
                 (' or combinations separated by dash like 240-365' if multi else '') +
                 ' (default: 240)')

    group.add_argument(
        '--ecm', '--ecms',
        type=str if multi else int,
        default=default,
        choices=['240', '365', '240-365', '365-240'] if multi else [240, 365],
        metavar=metavar,
        help=help_text
    )


def add_sel_argument(
    parser: ArgumentParser,
    default: str = '',
    multi: bool = False,
    group: str | None = None
) -> None:
    '''Add --sel argument for single selection strategy.'''
    if group is None:
        group: _ArgumentGroup = parser.add_argument_group('General arguments')
    if multi:
        group.add_argument(
            '--sels',
            type=str,
            default=default,
            help=f'Selections strategy to apply (dash-separated for multiple choices) (default: "{default}")'
        )
    else:
        group.add_argument(
            '--sel',
            type=str,
            default=default,
            help=f'Selection strategy to apply (default: "{default}")'
        )


def add_run_argument(
    parser: ArgumentParser,
    n_stages: int = 3,
    default: str = '2-3',
    group: str | None = None
) -> None:
    '''Add --run argument for pipeline stage selection.'''
    if n_stages == 2:
        choices = ['1', '2', '1-2']
    elif n_stages == 3:
        choices = ['1', '2', '3', '1-2', '2-3', '1-2-3']
    elif n_stages == 4:
        choices = ['1', '2', '3', '4', '1-2', '2-3',
                   '3-4', '1-2-3', '2-3-4', '1-2-3-4']
    else:
        raise ValueError(f'n_stages must be 2, 3, or 4, got {n_stages}')

    if group is None:
        group: ArgumentParser = parser.add_argument_group('Execution arguments')

    help_text = f'Pipeline stages to execute (1-{n_stages} or combinations separated by dash) (default: {default})'

    group.add_argument(
        '--run',
        type=str,
        default=default,
        choices=choices,
        metavar='STAGES',
        help=help_text
    )


def add_verbose_argument(
    parser: ArgumentParser,
    group: str | None = None
) -> None:
    '''Add -v/--verbose argument for debugging output.'''
    if group is None:
        group: _ArgumentGroup = parser.add_argument_group('General arguments')
    group.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output for debugging'
    )



# ============================================================== #
# FEATURE GROUP BUILDERS (higher-level, feature-specific groups) #
# ============================================================== #

######################################
### 1-MVAINPUTS SPECIFIC ARGUMENTS ###
######################################

#######################
## GENERAL ARGUMENTS ##
#######################

def add_selection_args(parser: ArgumentParser) -> None:
    '''Add selection options shared by MVA and measurement scripts.'''
    args = parser.add_argument_group('Selection arguments')
    args.add_argument(
        '--do-test',
        action=BooleanOptionalAction,
        default=False,
        help='Use the cut defined in the pre-selection'
    )


def add_sample_selection_args(parser: ArgumentParser) -> None:
    '''Add sample filtering options shared by selection scripts.'''
    args = parser.add_argument_group('Sample-selection arguments')
    args.add_argument(
        '--include',
        type=str,
        default='',
        help="Samples to include, separated by ':'; use sample,fraction,chunks for overrides"
    )
    args.add_argument(
        '--exclude',
        type=str,
        default='',
        help="Samples to exclude, separated by ':', or 'all' to exclude every sample"
    )



################################
## SCRIPTS SPECIFIC ARGUMENTS ##
################################

# 1-MVAInputs/pre-selection.py specific argument
def add_preselection_args(parser: ArgumentParser) -> None:
    '''Add execution options specific to pre-selection scripts.'''
    args = parser.add_argument_group('Pre-selection arguments')
    args.add_argument(
        '--job-flavor',
        type=str,
        default='workday',
        choices=['espresso', 'microcentury', 'longlunch',
                 'workday', 'tomorrow', 'testmatch', 'nextweek'],
        help='Job flavour for HTCondor (default: workday): '
        'espresso (20 min),'
        'microcentury (1 h), longlunch (2 h), workday (8 h),'
        'tomorrow (1 d), testmatch (3 d), nextweek (1 w)'
    )
    args.add_argument(
        '--run-batch',
        action='store_true',
        default=False,
        help='Submit jobs to HTCondor batch system'
    )

# 1-MVAInputs/final-selection.py specific argument
def add_final_selection_args(parser: ArgumentParser) -> None:
    '''Add options specific to final-selection scripts.'''
    args = parser.add_argument_group('Final-selection arguments')
    args.add_argument(
        '--do-tree',
        action='store_true',
        default=False,
        help='Save ROOT TTrees in addition to histograms (default: False)'
    )
    args.add_argument(
        '--do-sel0',
        action='store_true',
        default=False,
        help="Include 'No cut selection' in cutList"
    )


# 1-MVAInputs/plots.py specific arguments
def add_plot_args(
        parser: ArgumentParser,
        include_var: bool = True,
        include_sig_scale: bool = True
) -> None:
    '''Add options specific to MVA input plotting.'''
    args = parser.add_argument_group('Plot arguments')
    if include_var:
        args.add_argument(
            '--variables',
            type=str,
            default='all',
            help='Variables to plot (default: all)'
        )
    args.add_argument(
        '--formats',
        type=str,
        default='png',
        choices=['png', 'pdf', 'png-pdf', 'pdf-png'],
        help='Output file formats (default: png)'
    )
    args.add_argument(
        '--scale-sig',
        type=float,
        default=1.,
        help='Signal scaling in plots (default: 1)'
    )



# ==================================================================#
# FACTORY FUNCTIONS - Compose modular builders for specific scripts #
# ==================================================================#

def base_parser(
        description: str,
        include_cat: bool = False,
        cat_multi: bool = False,
        cat_default: str | None = None,
        allow_qq: bool = True,
        allow_empty: bool = False,
        include_ecm: bool = False,
        ecm_multi: bool = False,
        ecm_default: int | str = 240,
        include_sel: bool = False,
        sel_multi: bool = False,
        sel_default: str = '',
         ) -> ArgumentParser:
    '''Create the common parser shell used by script-specific builders.'''
    parser    = ArgumentParser(description=description)
    general   = parser.add_argument_group('General arguments')

    if include_cat:
        add_cat_argument(parser, cat_multi, allow_empty, cat_default, allow_qq, general)
    if include_ecm:
        add_ecm_argument(parser, ecm_multi, ecm_default, general)
    add_verbose_argument(parser, general)

    if include_sel:
        add_sel_argument(parser, sel_default, sel_multi, general)
    return parser



# ==================== #
# VALIDATION UTILITIES #
# ==================== #

def parse_args(
    parser: ArgumentParser,
    validate_cat: bool = False,
    comb: bool = False
) -> Namespace:
    '''
    Parse and validate command-line arguments.

    Args:
        parser: ArgumentParser instance
        validate_cat: Require --cat to be specified
        comb: For fit scripts - require either --cat or --combine

    Returns:
        Parsed arguments as Namespace

    Raises:
        SystemExit: If validation fails
    '''
    args = parser.parse_args()

    if comb and hasattr(args, 'combine'):
        if not (hasattr(args, 'cat') and args.cat) and not args.combine:
            parser.error('Either --cat or --combine must be specified for fit')
    elif validate_cat and (not hasattr(args, 'cat') or not args.cat):
        parser.error('--cat must be specified')

    return args



# ============= #
# LOGGING SETUP #
# ============= #

def set_log(args: Namespace | None) -> None:
    """
    Initialize logging system based on parsed arguments.

    level specified by the user (via -v/--verbose flag).

    This function should be called ONCE in your main analysis script,
    before any other imports that need logging.

    Parameters
    ----------
    args : Namespace
        Parsed arguments from parse_args()

    Examples
    --------
    In your main analysis script:

        from package.parsing import create_parser, parse_args, set_log
        from package.logger import get_logger

        # Parse arguments
        parser = create_parser(cat_single=True, include_sels=True)
        args = parse_args(parser)

        # Setup logging based on --verbose flag
        set_log(args)

        # Now you can use logging
        LOGGER = get_logger(__name__)
        LOGGER.info('Analysis starting')
    """
    from logger import setup_logging

    # Check if args has verbose flag
    verbose = getattr(args, 'verbose', False)

    # Initialize logging with the verbose flag
    setup_logging(verbose)
