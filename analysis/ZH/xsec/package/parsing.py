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

from argparse import ArgumentParser, Namespace


# ========================================================== #
# MODULAR ARGUMENT BUILDERS (low-level, reusable components) #
# ========================================================== #

def add_cat_argument(
        parser: ArgumentParser,
        multi: bool = False,
        allow_empty: bool = False,
        default: str = '',
        group=None
         ) -> None:
    '''Add --cat/--cats argument for final state selection.'''
    choices = ['ee', 'mumu', 'qq']
    if multi:
        choices.extend([
            'ee-mumu', 'ee-qq',   'ee-mumu-qq', 'ee-qq-mumu',
            'mumu-ee', 'mumu-qq', 'mumu-ee-qq', 'mumu-qq-ee',
            'qq-ee',   'qq-mumu', 'qq-ee-mumu', 'qq-mumu-ee'
        ])
    if allow_empty:
        choices.append('')

    if group is None:
        group = parser.add_argument_group('General arguments')

    # Use metavar to show a concise pattern instead of listing all choices
    metavar = 'CHANNELS' if multi else '{ee,mumu,qq}'
    help_text = ('Final state lepton category: ee, mumu, qq' +
                 (' or combinations separated by dash like ee-mumu' if multi else '') +
                 ' (default: "")')

    group.add_argument(
        '--cat', '--cats',
        type=str,
        default=default or ('ee-mumu' if multi else ''),
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

    metavar = 'ENERGIES' if multi else '{240,365}'
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
        help='Selections to process (dash-separated for multiple) (default: "")'
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

def add_bdt_eval(parser: ArgumentParser) -> None:
    '''Add BDT evaluation arguments (metric, tree, optionally check, hl).'''
    args = parser.add_argument_group('Evaluation arguments')
    args.add_argument(
        '--metric',
        action='store_true',
        default=False,
        help='Plot metric distributions'
    )
    args.add_argument(
        '--tree',
        action='store_true',
        default=False,
        help='Plot decision trees'
    )

    args.add_argument(
        '--check',
        action='store_true',
        default=False,
        help='Plot variable distributions'
    )
    args.add_argument(
        '--hl',
        action='store_true',
        default=False,
        help='Plot variable distributions for high/low score regions'
    )


def add_plots_args(parser: ArgumentParser) -> None:
    '''Add plots arguments (yields, decay, make, scan)'''
    args = parser.add_argument_group('Plots arguments')
    args.add_argument(
        '--yields',
        action='store_true',
        default=False,
        help='Do not make yields plots'
    )
    args.add_argument(
        '--decay',
        action='store_true',
        default=False,
        help='Do not make Higgs decays only plots'
    )
    args.add_argument(
        '--make',
        action='store_true',
        default=False,
        help='Do not make distribution plots',
    )
    args.add_argument(
        '--scan',
        action='store_true',
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


def add_optimize_args(parser: ArgumentParser) -> None:
    '''Add optimization arguments (procs, nevents, incr).'''
    args = parser.add_argument_group('Optimization arguments')
    args.add_argument(
        '--procs',
        type=str,
        default='',
        help='Processes to optimize for (comma-separated)'
    )
    args.add_argument(
        '--nevents',
        type=int,
        default=-1,
        help='Max events to process (-1: all) (default: -1)'
    )
    args.add_argument(
        '--incr',
        type=float,
        default=0.1,
        help='Parameter increment (default: 0.1)'
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
    '''Add fit arguments (pert, target, combine, bias, t, noprint).'''
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
        '-t', '--timer',
        action='store_true',
        default=False,
        dest='t',
        help='Display elapsed time'
    )
    args.add_argument(
        '--noprint',
        action='store_true',
        default=False,
        help='Suppress uncertainty output'
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
        ecm_multi: bool = False,
        ecm_default: int | str | None = None,
        allow_empty: bool = False,
        include_sel: bool = False,
        include_sels: bool = False,
        run_stages: int = 0,
        run_default: str = '2-3',
        batch: bool = False,
        bdt_eval: bool = False,
        plots: bool = False,
        cutflow: bool = False,
        optimize: bool = False,
        polarization: bool = False,
        fit: bool = False,
        bias: bool = False,
        bias_extra: bool = False,
        default_target: str = '',
        default_pert: float = 1.0,
        description: str = 'Analysis script'
         ) -> ArgumentParser:
    '''
    Factory function to compose a custom parser with selected argument groups.

    This single function replaces all the specific create_*_parser functions
    by allowing you to specify exactly which features you need.

    Args:
        cat_single: Add --cat for single channel
        cat_multi: Add --cat for multi-channel (ee-mumu combinations)
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
        add_cat_argument(parser, multi=cat_multi, allow_empty=allow_empty, default=cat_default, group=general)
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
        add_run_argument(parser, n_stages=run_stages, default=run_default, group=exec)
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
        add_optimize_args(parser)
    if polarization:
        add_polarization(parser)
    if fit:
        add_fit_args(parser, default_target=default_target, default_pert=default_pert)
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
