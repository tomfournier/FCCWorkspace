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
        '--test',
        action=BooleanOptionalAction,
        default=False,
        help='Use the cut defined in the pre-selection'
    )
    # Temporary argument
    args.add_argument(
        '--jan',
        action='store_true',
        default=False,
        help="Use Jan's definition of jets"
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
    args.add_argument(
        '--only-sig',
        action='store_true',
        default=False,
        help='Only do the pre-selection for the signal processes'
    )
    args.add_argument(
        '--only-bkg',
        action='store_true',
        default=False,
        help='Only do the pre-selection for the background processes'
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
        default='longlunch',
        choices=['espresso', 'microcentury', 'longlunch',
                 'workday', 'tomorrow', 'testmatch', 'nextweek'],
        help='Job flavour for HTCondor (default: longlunch): '
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
def add_final_selection_args(
        parser: ArgumentParser,
        training: bool = False
         ) -> None:
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
    if not training:
        args.add_argument(
            '--bdt-sel',
            type=str,
            default='',
            help="BDT selection to use, use nominal selection if '' (default: '')"
        )
        args.add_argument(
            '--weight-suffix',
            type=str,
            default='weights',
            help='BDT cut to use for high/low score region separation (default: weights)'
        )
        args.add_argument(
            '--hl-include',
            type=str,
            default='all',
            help='Selection to include for high/low separation (default: all)'
        )

# 1-MVAInputs/plots.py specific arguments
def add_mva_plot_args(parser: ArgumentParser) -> None:
    '''Add options specific to MVA input plotting.'''
    args = parser.add_argument_group('Plot arguments')
    args.add_argument(
        '--variable',
        type=str,
        default='all',
        help='Variables to plot (default: all)'
    )
    args.add_argument(
        '--exclusive-decays',
        action='store_true',
        default=False,
        help='Use wzp6_ee_xxH_Hyy_ecmECM samples instead of wzp6_ee_xxH_ecmECM'
    )
    args.add_argument(
        '--use-rare-bkgs',
        action='store_true',
        default=False,
        help="Include e gamma -> eZ and gaga -> ff processes into 'Rare' background"
    )
    args.add_argument(
        '--formats',
        type=str,
        default='png',
        choices=['png', 'pdf', 'png-pdf'],
        help='Output file formats (default: png)'
    )
    args.add_argument(
        '--scale-sig',
        type=float,
        default=1.,
        help='Signal scaling in plots (default: 1)'
    )



################################
### 2-BDT SPECIFIC ARGUMENTS ###
################################

def add_bdt_inputs(parser: ArgumentParser) -> None:
    '''Add BDT inputs script specific arguments'''
    args = parser.add_argument_group('BDT inputs arguments')
    args.add_argument(
        '--all-inputs',
        action=BooleanOptionalAction,
        default=True,
        help='Use all the events in each mode for the training (default True)'
    )
    args.add_argument(
        '--n-max',
        type=int,
        default=1_000_000,
        help='Maximum number of events per mode (default 1,000,000)'
    )

def add_bdt_training(parser: ArgumentParser) -> None:
    args = parser.add_argument_group('BDT training arguments')
    args.add_argument(
        '--use-weights',
        action='store_true',
        default=False,
        help='Use event weighting for BDT training (default False)'
    )
    args.add_argument(
        '--weight-name',
        type=str,
        default='train_weight',
        help="Name of the weight used for the BDT training (default 'train_weight')"
    )


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



########################################
### 3-MEASUREMENT SPECIFIC ARGUMENTS ###
########################################

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
    args.add_argument(
        '--hl',
        action=BooleanOptionalAction,
        default=True,
        help='Include <sel>_high/_low selection to plot'
    )
    args.add_argument(
        '--hlsel',
        type=str,
        default='Baseline-Baseline_miss-Baseline_sep-Baseline_vis-Baseline_inv-test',
        help='sels to include in the hl plot'
    )
    args.add_argument(
        '--hl-include',
        type=str,
        default='',
        help='Selections to include in the hl plots'
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
    args.add_argument(
        '--kin',
        action=BooleanOptionalAction,
        default=True,
        help='Use events from files with Baseline kinematic cuts already made (smaller files)'
    )



####################################
### 4-COMBINE SPECIFIC ARGUMENTS ###
####################################

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


def add_combine_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        '--mc-stats',
        action='store_true',
        default=False,
        help='Include MC statistical uncertainties (default False)'
    )
    parser.add_argument(
        '--rebin',
        type=int,
        default=1,
        help='Histogram rebinning factor (default 1)'
    )
    parser.add_argument(
        '--intLumi',
        type=float,
        default=1.,
        help='Luminosity scaling factor for normalization'
    )
    parser.add_argument(
        '--rescale',
        default=False,
        action='store_true',
        help='Rescale the histograms to 1 ab-1 (default False)'
    )



################################
### 5-FIT SPECIFIC ARGUMENTS ###
################################

#######################
## GENERAL ARGUMENTS ##
#######################

def add_fit_args(parser: ArgumentParser) -> None:
    args = parser.add_argument_group('Fit arguments')
    args.add_argument(
        '--lep', '--leptonic',
        action='store_true',
        default=False,
        help='Combine the fit for the ee and mumu channel. Do not use with --combine'
    )
    args.add_argument(
        '--combine', '--comb',
        action='store_true',
        default=False,
        help='Combine the fit the for all channel (ee, mumu, qq). Do not use with --lep'
    )
    args.add_argument(
        '--timer',
        action=BooleanOptionalAction,
        default=True,
        help='Display elapsed time'
    )
    args.add_argument(
        '--fastscan',
        action='store_true',
        default=False,
        help='Do a fast scan to check the fit'
    )
    args.add_argument(
        '--skip-setup',
        action='store_true',
        default=False,
        help='Skip the datacard and workspace setup'
    )
    args.add_argument(
        '--only-diag',
        action='store_true',
        default=False,
        help='Only do the diagnostic fit and skip the likelihood scan (decrease the fit precision)'
    )
    args.add_argument(
        '--fast-scan',
        action='store_true',
        default=False,
        help='Do a fast likelihood scan for the second fit'
    )
    args.add_argument(
        '--rescaled',
        default=False,
        action='store_true',
        help='Rescaled the uncertainties to the nominal luminosity (default False)'
    )
    args.add_argument(
        '--print',
        action=BooleanOptionalAction,
        default=True,
        help='Suppress uncertainty output'
    )


def add_bias_fit_args(parser: ArgumentParser, default_target: str = '') -> None:
    args = parser.add_argument_group('Fit arguments specific to bias test')
    args.add_argument(
        '--target',
        type=str,
        default=default_target,
        help=f'Target pseudodata (default: "{default_target}")'
    )
    args.add_argument(
        '--bias',
        action='store_true',
        default=False,
        help='Run bias test instead of nominal fit'
    )


###############################
## SCRIPT SPECIFIC ARGUMENTS ##
###############################

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
        help='Plot 95%% CL'
    )
    args.add_argument(
        '--only1',
        action='store_true',
        default=False,
        help='Only compute the scan for one target at a time'
    )
    args.add_argument(
        '--which',
        type=str,
        default='',
        choices=['', 'cat', 'ecm', 'sel', 'decay'],
        help='Choose which parameter to compare'
    )


def add_bias_args(parser: ArgumentParser,
                  default_pert: float = 1.05) -> None:
    '''Add bias test specific arguments (freeze, float, plot_dc).'''
    args = parser.add_argument_group('Bias test arguments')
    args.add_argument(
        '--pert',
        type=float,
        default=default_pert,
        help=f'Perturbation/scale factor (default: {default_pert})'
    )
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
    args.add_argument(
        '--plot-pseudo',
        action='store_true',
        default=False,
        help='Plot the pseudo-data distributions'
    )


def add_bias_extra_args(parser: ArgumentParser) -> None:
    '''Add extra options forwarded by the fit pipeline runner.'''
    parser.add_argument(
        '--extra',
        nargs='*',
        default=[],
        choices=['tot', 'onlyrun', 't'],
        help='Extra options for the fit pipeline'
    )


def add_nlo_args(parser: ArgumentParser) -> None:
    args = parser.add_argument_group('Self-coupling fit specific arguments')
    args.add_argument(
        '--model',
        type=str,
        default='SMEFT_Cphi_Cbox',
        help='Model used to parametrize the cross-section (default: SMEFT_Cphi_Cbox)'
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



##############################
### PARSER FOR 1-MVAINPUTS ###
##############################

# Parser for 1-MVAInputs/pre-selection.py
def mva_preselection_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Pre-selection Script',
        include_cat=True, include_ecm=True)
    add_selection_args(parser)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    return parser

# Parser for 1-MVAInputs/final-selection.py
def mva_final_selection_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Final-selection Script',
        include_cat=True, include_ecm=True,
        include_sel=True, sel_multi=True, sel_default='all')
    add_selection_args(parser)
    add_final_selection_args(parser, True)
    add_sample_selection_args(parser)
    return parser

# Parser for 1-MVAInputs/plots.py
def mva_plots_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Plot Script',
        include_cat=True, include_ecm=True,
        include_sel=True, sel_multi=True, sel_default='all')
    add_selection_args(parser)
    add_mva_plot_args(parser)
    return parser



########################
### PARSER FOR 2-BDT ###
########################

# Parser for 2-BDT/process_inputs.py
def bdt_process_input_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'BDT Input Processing Script',
        include_cat=True, include_ecm=True,
        include_sel=True, sel_multi=True)
    add_bdt_inputs(parser)
    return parser

# Parser for 2-BDT/train_bdt.py
def bdt_training_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'BDT Training Script',
        include_cat=True, include_ecm=True,
        include_sel=True, sel_multi=True)
    add_bdt_training(parser)
    return parser

# Parser for 2-BDT/evaluation.py
def bdt_evaluation_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'BDT Evaluation Script',
        include_cat=True, include_ecm=True,
        include_sel=True, sel_multi=True)
    add_bdt_eval(parser)
    return parser



################################
### PARSER FOR 3-MEASUREMENT ###
################################

# Parser for 3-Measurement/pre-selection.py
def measurement_preselection_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Pre-selection Script',
        include_cat=True, include_ecm=True)
    add_selection_args(parser)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    return parser

# Parser for 3-Measurement.final-selection.py
def measurement_final_selection_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Final-selection Script',
        include_cat=True, include_ecm=True,
        include_sel=True, sel_multi=True, sel_default='all')
    add_selection_args(parser)
    add_final_selection_args(parser)
    add_sample_selection_args(parser)
    return parser

# Parser for 3-Measurement/plots.py
def measurement_plots_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Measurement Plots Script',
        include_cat=True, cat_multi=True, include_ecm=True,
        include_sel=True, sel_multi=True)
    add_plots_args(parser)
    return parser

# Parser for 3-Measurement/cutflow.py
def measurement_cutflow_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Cutflow Script',
        include_cat=True, cat_multi=True, include_ecm=True,
        include_sel=True, sel_multi=True)
    add_cutflow_args(parser)
    return parser



############################
### PARSER FOR 4-COMBINE ###
############################

# Parser for 4-Combine/process_histogram.py
def combine_process_histogram_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Histogram Processing Script',
        include_cat=True, cat_multi=True, include_ecm=True,
        include_sel=True, sel_multi=True)
    add_polarization(parser)
    return parser

# Parser for 4-Combine/combine.py
def combine_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Datacard making script',
        include_cat=True, cat_multi=True,
        include_ecm=True, ecm_multi=True,
        include_sel=True, sel_multi=True)
    add_combine_args(parser)
    return parser



########################
### PARSER FOR 5-FIT ###
########################

# Parser for 5-Fit/fit.py
def fit_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Fit Script',
        include_cat=True, allow_empty=True, include_ecm=True,
        include_sel=True)
    add_fit_args(parser)
    add_bias_fit_args(parser)
    return parser

# Parser for 5-Fit/plots.py
def fit_plots_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Fit Plots Script',
        include_cat=True, cat_multi=True, cat_default='', allow_empty=True,
        include_ecm=True, ecm_multi=True,
        include_sel=True, sel_multi=True)
    add_fit_plot_args(parser)
    return parser

# Parser for 5-Fit.py/make_pseudo.py
def make_pseudo_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Pseudo-data Script',
        include_cat=True, allow_empty=True, include_ecm=True,
        include_sel=True)
    add_fit_args(parser)
    add_bias_fit_args(parser, 'bb')
    add_bias_args(parser, 1)
    add_polarization(parser)
    return parser

# Parser for 5-Fit/bias_test.py
def bias_test_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Bias Test Script',
        include_cat=True, allow_empty=True, include_ecm=True,
        include_sel=True)
    add_fit_args(parser)
    add_bias_fit_args(parser, 'bb')
    add_bias_args(parser, 1.05)
    add_polarization(parser)
    return parser



##################################
### PARSER FOR 6-SELF-COUPLING ###
##################################

# Parser for 6-Self-coupling/fit.py
def self_coupling_fit_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Fit Script',
        include_cat=True, allow_empty=True,
        include_sel=True)
    add_fit_args(parser)
    add_nlo_args(parser)
    return parser

# Parser for 6-Self-coupling/plots.py
def self_coupling_plots_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Fit Plots Script',
        include_cat=True, cat_multi=True, cat_default='', allow_empty=True,
        include_sel=True, sel_multi=True)
    add_fit_plot_args(parser)
    return parser



########################
### PARSER FOR 0-RUN ###
########################

# Parser for 0-Run/1-run.py
def MVAInputs_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run MVA Inputs pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_sel=True, sel_multi=True)
    add_run_argument(parser, 3)
    add_selection_args(parser)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    add_final_selection_args(parser, training=True)
    add_mva_plot_args(parser)
    return parser

# Parser for 0-Run/2-run.py
def BDT_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run BDT training pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_sel=True, sel_multi=True)
    add_run_argument(parser, 3)
    add_bdt_inputs(parser)
    add_bdt_training(parser)
    add_bdt_eval(parser)
    return parser

# Parser for 0-Run/3-run.py
def Measurement_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Measurement pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_sel=True, sel_multi=True)
    add_run_argument(parser, 4)
    add_selection_args(parser)
    add_preselection_args(parser)
    add_plots_args(parser)
    add_cutflow_args(parser)
    return parser

# Parser for 0-Run/4-run.py
def Combine_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Combine pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_sel=True, sel_multi=True)
    add_run_argument(parser, 2, default='1-2')
    add_polarization(parser)
    add_combine_args(parser)
    return parser

# Parser for 0-Run/5-run.py
def Fit_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Fit pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_sel=True, sel_multi=True)
    add_run_argument(parser, 2)
    add_fit_args(parser)
    add_bias_fit_args(parser, 'bb')
    add_bias_args(parser)
    add_bias_extra_args(parser)
    add_polarization(parser)
    return parser

# Parser for 0-Run/6-run.py
def NLO_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Self-coupling fit pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_sel=True, sel_multi=True)
    add_fit_args(parser)
    add_bias_fit_args(parser)
    add_fit_plot_args(parser)
    add_nlo_args(parser)
    return parser



##############################
### GLOBAL PARSER FUNCTION ###
##############################

def create_parser(
        directory: str,
        script: str | None = None,
        description: str | None = None
         ) -> ArgumentParser:
    '''Dispatch to a script parser or aggregate a whole directory parser.'''
    if script is None:
        if directory == '1-MVAInputs':   return MVAInputs_parser(description)
        if directory == '2-BDT':         return BDT_parser(description)
        if directory == '3-Measurement': return Measurement_parser(description)
        if directory == '4-Combine':     return Combine_parser(description)
        if directory == '5-Fit':         return Fit_parser(description)
        # Need to implement 6-Run.py
        # if directory == '6-Self-coupling': return NLO_parser(description)
        raise ValueError(f'Unknown parser directory: {directory}')

    # 1-MVAInputs directory
    if directory == '1-MVAInputs':
        if script == 'pre-selection':   return mva_preselection_parser(description)
        if script == 'final-selection': return mva_final_selection_parser(description)
        if script == 'plots':           return mva_plots_parser(description)

    # 2-BDT directory
    elif directory == '2-BDT':
        if script == 'process_input': return bdt_process_input_parser(description)
        if script == 'train_bdt':     return bdt_training_parser(description)
        if script == 'evaluation':    return bdt_evaluation_parser(description)

    # 3-Measurement directory
    elif directory == '3-Measurement':
        if script == 'pre-selection':   return measurement_preselection_parser(description)
        if script == 'final-selection': return measurement_final_selection_parser(description)
        if script == 'plots':           return measurement_plots_parser(description)
        if script == 'cutflow':         return measurement_cutflow_parser(description)

    # 4-Combine directory
    elif directory == '4-Combine':
        if script == 'process_histogram': return combine_process_histogram_parser(description)
        if script == 'combine':           return combine_parser(description)

    # 5-Fit directory
    elif directory == '5-Fit':
        if script == 'fit':         return fit_parser(description)
        if script == 'plots':       return fit_plots_parser(description)
        if script == 'make_pseudo': return make_pseudo_parser(description)
        if script == 'bias_test':   return bias_test_parser(description)

    # 6-Self-coupling directory
    elif directory == '6-Self-coupling':
        if script == 'fit':   return self_coupling_fit_parser(description)
        if script == 'plots': return self_coupling_plots_parser(description)

    raise ValueError(f'Unknown parser script: {directory}/{script}')



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
    from package.logger import setup_logging

    # Check if args has verbose flag
    verbose = getattr(args, 'verbose', False)

    # Initialize logging with the verbose flag
    setup_logging(verbose)
