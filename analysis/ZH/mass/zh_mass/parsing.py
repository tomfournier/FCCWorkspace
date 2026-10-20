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

from argparse import ArgumentParser, BooleanOptionalAction

from parsing import (
    add_plot_args,
    base_parser,
    add_run_argument,
    add_selection_args,
    add_preselection_args,
    add_final_selection_args,
    add_sample_selection_args
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

def add_xsec_sample_selection_args(parser: ArgumentParser) -> None:
    '''Add sample filtering options shared by selection scripts.'''
    args = parser.add_argument_group('Sample-selection arguments')
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



########################################
### 1-SELECTION SPECIFIC ARGUMENTS ###
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
        help='Include MC statistical uncertainties (default: False)'
    )
    parser.add_argument(
        '--rebin',
        type=int,
        default=1,
        help='Histogram rebinning factor (default: 1)'
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
        help='Rescale the histograms to 1 ab-1 (default: False)'
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
        '--npoints',
        type=int,
        default=400,
        help='Number of points to do for the likelihood scan (default: 400)'
    )
    args.add_argument(
        '--rescaled',
        default=False,
        action='store_true',
        help='Rescaled the uncertainties to the nominal luminosity (default: False)'
    )
    args.add_argument(
        '--print',
        action=BooleanOptionalAction,
        default=True,
        help='Suppress uncertainty output'
    )

def add_mass_fit_args(parser: ArgumentParser) -> None:
    '''Add mass fit arguments (pert, target, combine, bias, t, print).'''
    args = parser.add_argument_group('Higgs mass fit arguments')
    args.add_argument(
        '--recoilMin',
        type=float,
        default=120,
        help='Lower mass limit of the fit (default: 120)'
    )
    args.add_argument(
        '--recoilMax',
        type=float,
        default=140,
        help='Higher mass limit of the fit (default: 140)'
    )
    args.add_argument(
        '--hName',
        type=str,
        default='zll_recoil_m',
        help='Name of the histogram to fit (default: zll_recoil_m)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='Detector mode',
        choices=['IDEA', 'IDEA_MC', 'IDEA_3T', 'CLD', 'CLD_FullSim',
                 'IDEA_noBES', 'IDEA_2E', 'IDEA_BES6pct'],
        default='IDEA'
    )
    parser.add_argument(
        '--category',
        type=int,
        help='Category (0, 1, 2 or 3) (default 0)',
        choices=[0, 1, 2, 3],
        default=0
    )
    parser.add_argument(
        '--tag',
        type=str,
        help='Analysis tag for versioning, optional',
        default=''
    )
    parser.add_argument(
        '--nBins',
        type=int,
        help='Number of bins for plotting (default 250)',
        default=250
    )
    parser.add_argument(
        '--normYield',
        type=bool,
        help='Normalize the histograms (default True)',
        action=BooleanOptionalAction,
        default=True
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
        '--combine', '--comb',
        action='store_true',
        default=False,
        help='Plot the combined fit'
    )
    args.add_argument(
        '--timer',
        action=BooleanOptionalAction,
        default=True,
        help='Display elapsed time'
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
        choices=['', 'cat', 'ecm', 'sel'],
        help='Choose which parameter to compare'
    )


def add_datacard_args(parser: ArgumentParser) -> None:
    '''Add mass fit arguments (pert, target, combine, bias, t, print).'''
    args = parser.add_argument_group('Datacard making arguments')
    args.add_argument(
        '--syst',
        type=bool,
        action=BooleanOptionalAction,
        default=True,
        help='Include the systematic uncertainties in the fit (default True)'
    )

def add_fit_manager_args(parser: ArgumentParser) -> None:
    args = parser.add_argument_group('Fit manager arguments')
    args.add_argument(
        '--statOnly',
        action='store_true',
        default=False,
        help='Do stat-only fit'
    )
    args.add_argument(
        '--doSummary',
        action='store_true',
        default=False,
        help='Do summary plots'
    )
    args.add_argument(
        '--freezeBkg',
        action='store_true',
        default=False,
        help='Freeze the background for the fit'
    )
    args.add_argument(
        '--noBkg',
        action='store_true',
        default=False,
        help='Remove the background for the fit'
    )
    args.add_argument(
        '--doBreakdown',
        action='store_true',
        default=False,
        help='Do breakdown of the nuisance parameters'
    )
    args.add_argument(
        '--combine',
        action='store_true',
        default=False,
        help='Do the full combination fit'
    )


# ==================================================================#
# FACTORY FUNCTIONS - Compose modular builders for specific scripts #
# ==================================================================#


################################
### PARSER FOR 1-SELECTION ###
################################

# Parser for 3-Selection/pre-selection.py
def measurement_preselection_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Pre-selection Script',
        include_cat=True, include_ecm=True)
    add_selection_args(parser)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    add_xsec_sample_selection_args(parser)
    return parser

# Parser for 3-Measurement/final-selection.py
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
        description or 'Selection Plots Script',
        include_cat=True, cat_multi=True, include_ecm=True,
        include_sel=True, sel_multi=True, sel_default='Baseline')
    add_plot_args(parser)
    add_plots_args(parser)
    return parser

# Parser for 3-Measurement/cutflow.py
def measurement_cutflow_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Cutflow Script',
        include_cat=True, include_ecm=True,
        include_sel=True, sel_multi=True, sel_default='Baseline')
    add_plot_args(parser)
    add_cutflow_args(parser)
    return parser



########################
### PARSER FOR 2-FIT ###
########################

# Parser for 2-Fit/process_histogram.py
def combine_process_histogram_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Histogram Processing Script',
        include_cat=True, cat_multi=True, include_ecm=True,
        include_sel=True, sel_multi=True, sel_default='Baseline')
    add_polarization(parser)
    return parser

# Parser for 2-fit/combine.py
def combine_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Datacard making script',
        include_cat=True, cat_multi=True,
        include_ecm=True, ecm_multi=True,
        include_sel=True, sel_multi=True)
    add_combine_args(parser)
    return parser


# Parser for 2-Fit/fit.py
def fit_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Fit Script',
        include_cat=True, allow_empty=True, include_ecm=True,
        include_sel=True)
    add_fit_args(parser)
    return parser

# Parser for 2-Fit/plots.py
def fit_plots_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Fit Plots Script',
        include_cat=True, cat_multi=True, cat_default='', allow_empty=True,
        include_ecm=True, ecm_multi=True,
        include_sel=True, sel_multi=True)
    add_plot_args(parser, False, False)
    add_fit_plot_args(parser)
    return parser



########################
### PARSER FOR 3-JAN ###
########################

# Parser for 3-Jan/parametrize_signal.py
def parametrize_signal_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Signal Parametrization Script',
        include_cat=True, allow_qq=False,
        include_ecm=True, include_sel=True)
    add_mass_fit_args(parser)
    return parser

# Parser for 3-Jan/make_datacard.py
def make_datacard_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Datacard Making Script',
        include_cat=True, allow_qq=False,
        include_ecm=True, include_sel=True)
    add_mass_fit_args(parser)
    add_datacard_args(parser)
    return parser

# Parser for 3-Jan/fit_manager.py
def fit_manager_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Fit Manager Script',
        include_cat=True, allow_qq=False,
        include_ecm=True, include_sel=True)
    add_mass_fit_args(parser)
    add_fit_manager_args(parser)
    return parser


########################
### PARSER FOR 0-RUN ###
########################

# Parser for 0-Run/1-run.py
def Measurement_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Selection pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_ecm=True, ecm_multi=True,
                         include_sel=True, sel_multi=True, sel_default='all')
    add_run_argument(parser, 4)
    add_selection_args(parser)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    add_xsec_sample_selection_args(parser)
    add_final_selection_args(parser)
    add_plot_args(parser)
    add_plots_args(parser)
    add_cutflow_args(parser)
    return parser

# Parser for 0-Run/2-run.py
def Fit_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Combine and fit pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_ecm=True, ecm_multi=True,
                         include_sel=True, sel_multi=True)
    add_run_argument(parser, 3, '1-2')
    add_polarization(parser)
    add_combine_args(parser)
    add_fit_args(parser)
    add_polarization(parser)
    return parser

# Parser ofr 0-Run/3-run.py
def Jan_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Mass fit pipeline',
                         include_cat=True, cat_multi=True, cat_default='ee-mumu',
                         include_ecm=True, ecm_multi=True,
                         include_sel=True, sel_multi=True)
    add_run_argument(parser, 3, '1-2')
    add_mass_fit_args(parser)
    add_datacard_args(parser)
    add_fit_manager_args(parser)
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
        if directory == '1-Selection': return Measurement_parser(description)
        if directory == '2-Fit':       return Fit_parser(description)
        if directory == '3-Jan':       return Jan_parser(description)
        raise ValueError(f'Unknown parser directory: {directory}')

    # 1-Selection directory
    elif directory == '1-Selection':
        if script == 'pre-selection':   return measurement_preselection_parser(description)
        if script == 'final-selection': return measurement_final_selection_parser(description)
        if script == 'plots':           return measurement_plots_parser(description)
        if script == 'cutflow':         return measurement_cutflow_parser(description)

    # 2-Fit directory
    elif directory == '2-Fit':
        if script == 'process_histogram': return combine_process_histogram_parser(description)
        if script == 'combine':           return combine_parser(description)
        if script == 'fit':               return fit_parser(description)
        if script == 'plots':             return fit_plots_parser(description)

    elif directory == '3-Jan':
        if script == 'parametrize_signal': return parametrize_signal_parser(description)
        if script == 'make_datacard':      return make_datacard_parser(description)
        if script == 'fit_manager':        return fit_manager_parser(description)

    raise ValueError(f'Unknown parser script: {directory}/{script}')
