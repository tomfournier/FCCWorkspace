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
    Basic setup with just core arguments:
        from package.parsing import create_parser
        parser = create_parser(cat_multi=True, ecm_multi=True)
        args = parser.parse_args()

    With evaluations for BDT:
        parser = create_parser(
            cat_multi=True, ecm_multi=True,
            include_sels=True,           # For batch selection processing
            bdt_eval=True,               # For metric/tree plotting
            bdt_extra=True               # For check/hl options
        )
        args = parser.parse_args()
'''

from argparse import ArgumentParser, Namespace, BooleanOptionalAction


# ========================================================== #
# MODULAR ARGUMENT BUILDERS (low-level, reusable components) #
# ========================================================== #

def add_cat_argument(
        parser: ArgumentParser,
        multi: bool = False,
        allow_empty: bool = False,
        default: str = '',
        allow_qq: bool = True,
        group=None
         ) -> None:
    '''Add --cat/--cats argument for final state selection.'''

    def_value = default or ('ee-mumu' if multi else '')

    choices = ['ee', 'mumu', 'qq'] if allow_qq else ['ee', 'mumu']
    if multi:
        choices.extend(['ee-mumu', 'mumu-ee'])
        if allow_qq:
            choices.extend([
                'ee-qq', 'ee-mumu-qq', 'ee-qq-mumu',
                'mumu-qq', 'mumu-ee-qq', 'mumu-qq-ee',
                'qq-ee', 'qq-mumu', 'qq-ee-mumu', 'qq-mumu-ee'
            ])
    if allow_empty:
        choices.append('')

    if group is None:
        group = parser.add_argument_group('General arguments')

    # Use metavar to show a concise pattern instead of listing all choices
    metavar = 'CHANNELS' if multi else ('{ee, mumu, qq}' if allow_qq else '{ee, mumu}')
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
        default=None,
        group=None
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
        group = parser.add_argument_group('General arguments')

    metavar = 'ENERGIES' if multi else '{240, 365}'
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
        group=None
         ) -> None:
    '''Add --sel argument for single selection strategy.'''
    if group is None:
        group = parser.add_argument_group('General arguments')
    group.add_argument(
        '--sel',
        type=str,
        default=default,
        help=f'Selection strategy to apply (default: "{default}")'
    )


def add_sels_argument(
        parser: ArgumentParser,
        default: str = '',
        group=None
         ) -> None:
    '''Add --sels argument for multiple selection strategies (batch mode).'''
    if group is None:
        group = parser.add_argument_group('General arguments')
    group.add_argument(
        '--sels',
        type=str,
        default=default,
        help='Selections to process (dash-separated for multiple choices) (default: "")'
    )


def add_run_argument(
        parser: ArgumentParser,
        n_stages: int = 3,
        default: str = '2-3',
        group=None
         ) -> None:
    '''Add --run argument for pipeline stage selection.'''
    if n_stages == 2:
        choices = ['1', '2', '1-2']
    elif n_stages == 3:
        choices = ['1', '2', '3', '1-2', '2-3', '1-2-3']
    elif n_stages == 4:
        choices = ['1', '2', '3', '4', '1-2', '2-3', '3-4', '1-2-3', '2-3-4', '1-2-3-4']
    else:
        raise ValueError(f'n_stages must be 2, 3, or 4, got {n_stages}')

    if group is None:
        group = parser.add_argument_group('Execution arguments')

    help_text = f'Pipeline stages to execute (1-{n_stages} or combinations separated by dash) (default: {default})'

    group.add_argument(
        '--run',
        type=str,
        default=default,
        choices=choices,
        metavar='STAGES',
        help=help_text
    )


def add_verbose_argument(parser: ArgumentParser, group=None) -> None:
    '''Add -v/--verbose argument for debugging output.'''
    if group is None:
        group = parser.add_argument_group('General arguments')
    group.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output for debugging'
    )


def add_batch_argument(parser: ArgumentParser, group=None) -> None:
    '''Add --batch for HTCondor batch processing.'''
    if group is None:
        group = parser.add_argument_group('Execution arguments')
    group.add_argument(
        '--batch',
        action='store_true',
        default=False,
        help='Submit jobs to HTCondor batch system'
    )


# ============================================================== #
# FEATURE GROUP BUILDERS (higher-level, feature-specific groups) #
# ============================================================== #

def add_plots_args(parser: ArgumentParser) -> None:
    '''Add plots arguments (yields, decay, make, scan)'''
    args = parser.add_argument_group('Plots arguments')
    args.add_argument(
        '--yields',
        action=BooleanOptionalAction,
        default=True,
        help='Make yields plots'
    )
    args.add_argument(
        '--make',
        action=BooleanOptionalAction,
        default=True,
        help='Make distribution plots',
    )
    args.add_argument(
        '--scan',
        action=BooleanOptionalAction,
        default=False,
        help='Make significance scan plots',
    )


def add_cutflow_args(parser: ArgumentParser) -> None:
    '''Add cutflow argument (tot)'''
    args = parser.add_argument_group('Cutflow arguments')
    args.add_argument(
        '--tot',
        action='store_true',
        default=False,
        help='Include all the Z decays in the cutflow'
    )

def add_polarization(parser: ArgumentParser) -> None:
    '''Add polarization and luminosity scaling arguments.'''
    args = parser.add_argument_group('Polarization arguments')
    args.add_argument(
        '--polL',
        action='store_true',
        default=False,
        help='Scale to left polarization'
    )
    args.add_argument(
        '--polR',
        action='store_true',
        default=False,
        help='Scale to right polarization'
    )
    args.add_argument(
        '--ILC',
        action='store_true',
        default=False,
        help='Scale to ILC luminosity'
    )


def add_fit_args(
        parser: ArgumentParser,
         ) -> None:
    '''Add fit arguments (pert, target, combine, bias, t, print).'''
    args = parser.add_argument_group('Fit arguments')
    args.add_argument(
        '--combine', '--comb',
        action='store_true',
        default=False,
        help='Combine channels for fit'
    )
    args.add_argument(
        '--timer',
        action=BooleanOptionalAction,
        default=True,
        help='Display elapsed time'
    )
    args.add_argument(
        '--print',
        action=BooleanOptionalAction,
        default=True,
        help='Suppress uncertainty output'
    )


def add_mass_fit_args(
        parser: ArgumentParser,
         ) -> None:
    '''Add mass fit arguments (pert, target, combine, bias, t, print).'''
    args = parser.add_argument_group('Fit arguments')
    args.add_argument(
        '--low',
        type=float,
        default=120,
        help='Lower limit of the fit (default: 120)'
    )
    args.add_argument(
        '--high',
        type=float,
        default=140,
        help='Higher limit of the fit (default: 140)'
    )
    args.add_argument(
        '--hname',
        type=float,
        default='zll_recoil_m',
        help='Name of the histogram to fit (default: zll_recoil_m)'
    )



# ============================================================================
# FACTORY FUNCTIONS - Compose modular builders for specific scripts
# ============================================================================

def create_parser(
        cat_single: bool = False,
        cat_multi: bool = False,
        cat_default: str = '',
        allow_qq: bool = True,
        ecm_multi: bool = False,
        ecm_default: int | str | None = None,
        allow_empty: bool = False,
        include_sel: bool = False,
        include_sels: bool = False,
        run_stages: int = 0,
        run_default: str = '2-3',
        batch: bool = False,
        plots: bool = False,
        cutflow: bool = False,
        polarization: bool = False,
        fit: bool = False,
        mass_fit: bool = False,
        description: str = 'Analysis script'
         ) -> ArgumentParser:
    '''
    Factory function to compose a custom parser with selected argument groups.

    This single function replaces all the specific create_*_parser functions
    by allowing you to specify exactly which features you need.

    Args:
        cat_single: Add --cat for single channel
        cat_multi: Add --cat for multi-channel (ee-mumu combinations)
        cat_default: Default value for --cat (None: auto-sets to 'ee-mumu' if cat_multi=True, else '')
        ecm_multi: Add --ecm for multi-energy (240-365 combinations)
        allow_empty_cat: Allow empty string for --cat
        include_sel: Add --sel (single selection)
        include_sels: Add --sels (batch selections)
        run_stages: Add --run with N stages (0=none, 2/3/4 for pipeline stages)
        batch: Add --batch for HTCondor
        bdt_eval: Add basic BDT eval args (metric, tree)
        bdt_extra: Add extended BDT args (check, hl) - requires bdt_eval=True
        optimize: Add optimization args (procs, nevents, incr)
        polarization: Add --polL, --polR, --ILC
        fit: Add fit arguments
        bias: Add bias test arguments - requires fit=True
        default_target: Default value for --target
        default_pert: Default value for --pert
        description: Parser description

    Returns:
        Configured ArgumentParser ready to use
    '''
    parser = ArgumentParser(description=description)

    # Create argument groups once to avoid duplication in help output
    general = parser.add_argument_group('General arguments')
    exec    = parser.add_argument_group('Execution arguments')

    # Core arguments (share the same groups)
    if cat_single or cat_multi:
        add_cat_argument(parser, multi=cat_multi, allow_empty=allow_empty, default=cat_default, allow_qq=allow_qq, group=general)
    if ecm_multi or (cat_single or cat_multi):
        add_ecm_argument(parser, multi=ecm_multi, default=ecm_default, group=general)
    add_verbose_argument(parser, group=general)

    # Selection arguments (share the same group)
    if include_sel:
        add_sel_argument(parser, default='Baseline' if fit else '', group=general)
    if include_sels:
        add_sels_argument(parser, default='', group=general)

    # Execution arguments (share the same group)
    if run_stages > 0:
        add_run_argument(parser, n_stages=run_stages, default=run_default, group=exec)
    if batch:
        add_batch_argument(parser, group=exec)

    # Feature groups
    if plots:
        add_plots_args(parser)
    if cutflow:
        add_cutflow_args(parser)
    if polarization:
        add_polarization(parser)
    if fit:
        add_fit_args(parser)
    if mass_fit:
        add_mass_fit_args(parser)

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


# Keep legacy convenience function for backwards compatibility
def include_polarizations(parser: ArgumentParser) -> None:
    '''Legacy function - use add_polarization() instead.'''
    add_polarization(parser)


# ==================== #
# LOGGING SETUP        #
# ==================== #

def set_log(args: Namespace) -> None:
    """
    Initialize logging system based on parsed arguments.

    Call this right after parse_args() to configure logging with the verbosity
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

        from package.parsing import create_parser, parse_args, setup_logger_from_args
        from package.logger import get_logger

        # Parse arguments
        parser = create_parser(cat_single=True, include_sels=True)
        args = parse_args(parser)

        # Setup logging based on --verbose flag
        setup_logger_from_args(args)

        # Now you can use logging
        LOGGER = get_logger(__name__)
        LOGGER.info('Analysis starting')
    """
    from package.logger import setup_logging

    # Check if args has verbose flag
    verbose = getattr(args, 'verbose', False)

    # Initialize logging with the verbose flag
    setup_logging(verbose=verbose)
