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

    With BDT evaluation and selection processing:
        parser = create_parser(
            cat_multi=True, ecm_multi=True,
            include_sels=True,           # For batch selection processing
            bdt_eval=True                # For metric and tree plotting
        )
        args = parser.parse_args()

    With optimization and fitting:
        parser = create_parser(
            cat_single=True,
            optimize=True,               # For optimization arguments
            fit=True, bias=True          # For fitting and bias tests
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
        group: str | None = None
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
        group: str | None = None
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
        add_test: bool = False,
        group: str | None = None
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
    if add_test:
        group.add_argument(
            '--test',
            action=BooleanOptionalAction,
            default=True,
            help='Set test to True for pre-selection'
        )


def add_verbose_argument(
        parser: ArgumentParser,
        group: str | None = None
         ) -> None:
    '''Add -v/--verbose argument for debugging output.'''
    if group is None:
        group = parser.add_argument_group('General arguments')
    group.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output for debugging'
    )


def add_batch_argument(
        parser: ArgumentParser,
        group: str | None = None
         ) -> None:
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

def add_bdt_eval(parser: ArgumentParser) -> None:
    '''Add BDT evaluation arguments (metric, tree, optionally check, hl).'''
    args = parser.add_argument_group('Evaluation arguments')
    args.add_argument(
        '--metric',
        action=BooleanOptionalAction,
        default=True,
        help='Plot metric distributions'
    )
    args.add_argument(
        '--tree',
        action=BooleanOptionalAction,
        default=False,
        help='Plot decision trees'
    )

    args.add_argument(
        '--check',
        action=BooleanOptionalAction,
        default=False,
        help='Plot variable distributions'
    )
    args.add_argument(
        '--hl',
        action=BooleanOptionalAction,
        default=False,
        help='Plot variable distributions for high/low score regions'
    )


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
        '--decay',
        action=BooleanOptionalAction,
        default=True,
        help='Make Higgs decays only plots'
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
        action=BooleanOptionalAction,
        default=True,
        help='Include all the Z decays in the cutflow'
    )
    # args.add_argument(
    #     '--test',
    #     action=BooleanOptionalAction,
    #     default=True,
    #     help='Use events from files with kinematic cuts made (smaller files)'
    # )


def add_optimize_args(
        parser: ArgumentParser,
        is_plot: bool = False,
        is_run: bool = False,
        only_procs: bool = False
         ) -> None:
    '''Add optimization arguments (procs, method, nevents, incr, metric, dist).'''
    args = parser.add_argument_group('Optimization arguments')
    args.add_argument(
        '--procs',
        type=str,
        default='',
        help='Processes to optimize for (comma-separated)'
    )
    args.add_argument(
        '--method',
        '--methods',
        type=str,
        default='mll-pll',
        choices=['mll', 'pll', 'mll-pll', 'pll-mll'],
        metavar='METHODS',
        help="chi2 method to use (default: 'mll-pll')"
    )
    if (not is_plot or is_run) and not only_procs:
        args.add_argument(
            '--nevents',
            type=int,
            default=-1,
            help='Max events to process (-1: all) (default: -1)'
        )
        args.add_argument(
            '--incr',
            type=float,
            default=0.01,
            help='Parameter increment (default: 0.1)'
        )
    if (is_plot or is_run) and not only_procs:
        parser.add_argument(
            '--metrics',
            action=BooleanOptionalAction,
            default=True,
            help='Plot the optimisation metrics'
        )
        parser.add_argument(
            '--dist',
            action=BooleanOptionalAction,
            default=True,
            help='Plot the variables distribution'
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
        default_target: str = '',
        default_pert: float = 1.0
         ) -> None:
    '''Add fit arguments (pert, target, combine, bias, timer, print).'''
    args = parser.add_argument_group('Fit arguments')
    args.add_argument(
        '--pert',
        type=float,
        default=default_pert,
        help=f'Perturbation/scale factor (default: {default_pert})'
    )
    args.add_argument(
        '--target',
        type=str,
        default=default_target,
        help=f'Target pseudodata (default: "{default_target}")'
    )
    args.add_argument(
        '--combine', '--comb',
        action='store_true',
        default=False,
        help='Combine channels for fit'
    )
    args.add_argument(
        '--bias',
        action='store_true',
        default=False,
        help='Run bias test instead of nominal fit'
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

def add_fit_plot_args(
        parser: ArgumentParser,
         ) -> None:
    args = parser.add_argument_group('Fit plot arguments')
    args.add_argument(
        "--param",
        default="r",
        help="Parameter to scan (default: r)"
    )
    args.add_argument(
        "--y-cut",
        type=float,
        default=7.0,
        help="Remove points with y > y-cut"
    )
    args.add_argument(
        "--y-max",
        type=float,
        default=-1,
        help="Y-axis maximum"
    )
    args.add_argument(
        '--sig2',
        action='store_true',
        help='Plot 95% CL'
    )
    args.add_argument(
        '--bias',
        action='store_true',
        help='Do likelyhood scan for bias fit'
    )
    args.add_argument(
        '--comb',
        action='store_true',
        help='Do the likelyhood scan for combine fit'
    )
    args.add_argument(
        '--which',
        type=str,
        default='',
        choices=['', 'cat', 'ecm', 'sel', 'decay'],
        help='Choose which parameter to compare'
    )


def add_bias_args(parser: ArgumentParser, extra: bool = False) -> None:
    '''Add bias test specific arguments (freeze, float, plot_dc).'''
    args = parser.add_argument_group('Bias test arguments')
    args.add_argument(
        '--freeze',
        action='store_true',
        default=False,
        help='Freeze background parameters'
    )
    args.add_argument(
        '--float',
        action='store_true',
        default=False,
        help='Float background parameters'
    )
    args.add_argument(
        '--plot_dc',
        action='store_true',
        default=False,
        help='Plot datacard contents'
    )

    if extra:
        parser.add_argument(
            '--extra',
            nargs='*',
            default=[],
            choices=['tot', 'onlyrun', 't'],
            help='Extra argument for the fit',
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
        add_test: bool = False,
        batch: bool = False,
        bdt_eval: bool = False,
        plots: bool = False,
        cutflow: bool = False,
        optimize: bool = False,
        is_plot: bool = False,
        is_run: bool = False,
        only_procs: bool = False,
        polarization: bool = False,
        fit: bool = False,
        fit_plot: bool = False,
        bias: bool = False,
        bias_extra: bool = False,
        default_target: str = '',
        default_pert: float = 1.0,
        description: str = 'Analysis script'
         ) -> ArgumentParser:
    '''
    Factory function to compose a custom parser with selected argument groups.

    This replaces specific create_*_parser functions by letting you specify
    exactly which features your script needs. Mix and match argument groups
    to build flexible parsers for different analysis stages.

    Args:
        cat_single: Add --cat for single lepton channel (ee, mumu, or qq)
        cat_multi: Add --cat for multi-channel mode (ee-mumu combinations)
        cat_default: Default channel value (auto-sets to 'ee-mumu' if cat_multi=True)
        allow_qq: Include hadronic qq channel in choices (default: True)
        allow_empty: Allow empty string for --cat (default: False)
        ecm_multi: Add --ecm for multi-energy mode (240-365 combinations)
        ecm_default: Default energy value (default: '240' if ecm_multi, else 240)
        include_sel: Add --sel for single selection strategy
        include_sels: Add --sels for batch processing multiple selections
        run_stages: Add --run for pipeline execution (0=none, 2/3/4 stages available)
        run_default: Default pipeline stages (default: '2-3')
        add_test: Add --test flag for pre-selection testing
        batch: Add --batch for HTCondor job submission
        bdt_eval: Add BDT evaluation arguments (--metric, --tree, --check, --hl)
        plots: Add plotting arguments (--yields, --decay, --make, --scan)
        cutflow: Add cutflow argument (--tot for all Z decays)
        optimize: Add optimization arguments (--procs, --method, --nevents, --incr)
        is_plot: Context flag for optimization plotting (used internally)
        is_run: Context flag for optimization running (used internally)
        only_procs: Only include --procs argument for optimization
        polarization: Add polarization arguments (--polL, --polR, --ILC)
        fit: Add fit arguments (--pert, --target, --combine, --bias, --timer, --print)
        bias: Add bias test arguments (--freeze, --float, --plot_dc); requires fit=True
        bias_extra: Add --extra argument for bias tests with choices ['tot', 'onlyrun', 't']
        default_target: Default pseudodata target (default: '')
        default_pert: Default perturbation scale factor (default: 1.0)
        description: Parser description shown in help

    Returns:
        Configured ArgumentParser ready for parse_args()

    Notes:
        - Arguments are organized into logical groups for clean help output
        - Selection arguments (--sel, --sels) share a group
        - Execution arguments (--run, --batch) share a group
        - Each feature group (BDT, plots, fit, etc.) has its own section
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
        add_sel_argument(parser, default='Baseline' if fit or bias else '', group=general)
    if include_sels:
        add_sels_argument(parser, default='', group=general)

    # Execution arguments (share the same group)
    if run_stages > 0:
        add_run_argument(parser, n_stages=run_stages, default=run_default, add_test=add_test, group=exec)
    if batch:
        add_batch_argument(parser, group=exec)

    # Feature groups
    if bdt_eval:
        add_bdt_eval(parser)
    if plots:
        add_plots_args(parser)
    if cutflow:
        add_cutflow_args(parser)
    if optimize:
        add_optimize_args(parser, is_plot, is_run, only_procs)
    if polarization:
        add_polarization(parser)
    if fit:
        add_fit_args(parser, default_target=default_target, default_pert=default_pert)
    if fit_plot:
        add_fit_plot_args(parser)
    if bias and fit:
        add_bias_args(parser, extra=bias_extra)

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
    from package.logger import setup_logging

    # Check if args has verbose flag
    verbose = getattr(args, 'verbose', False)

    # Initialize logging with the verbose flag
    setup_logging(verbose=verbose)
