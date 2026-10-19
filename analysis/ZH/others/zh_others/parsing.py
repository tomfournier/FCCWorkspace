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

from argparse import ArgumentParser, BooleanOptionalAction

from parsing import (
    base_parser,
    add_plot_args,
    add_run_argument,
    add_preselection_args,
    add_sample_selection_args
)

# ============================================================== #
# FEATURE GROUP BUILDERS (higher-level, feature-specific groups) #
# ============================================================== #

######################################
### OPTIMIZATION SCRIPTS ARGUMENTS ###
######################################

def add_procs_args(parser: ArgumentParser) -> None:
    '''Add process selection arguments'''
    args = parser.add_argument_group('process selection arguments')
    args.add_argument(
        '--procs',
        type=str,
        default='',
        help='Processes to optimize for (comma-separated)'
    )



#################################
### b-CHI2 SPECIFIC ARGUMENTS ###
#################################

def add_optimize_args(parser: ArgumentParser) -> None:
    '''Add chi2 optimization arguments.'''
    args = parser.add_argument_group('Chi2 optimization arguments')
    args.add_argument(
        '--methods',
        type=str,
        default='mll-pll',
        choices=['mll', 'pll', 'mll-pll', 'pll-mll'],
        metavar='METHODS',
        help="chi2 method to use (default: 'mll-pll')"
    )

def add_chi2_optimization_args(parser: ArgumentParser) -> None:
    args = parser.add_argument_group('Chi2 Optimization arguments')
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

def add_plots_chi2_args(parser: ArgumentParser) -> None:
    '''Add chi2 plots arguments.'''
    args = parser.add_argument_group('Chi2 plots arguments')
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



# ================================================================= #
# FACTORY FUNCTIONS - Compose modular builders for specific scripts #
# ================================================================= #

########################
### PARSER FOR a-FSR ###
########################

# Parser for a-FSR/pre-selection.py
def fsr_preselection_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'FSR recovery Pre-selection Script',
        include_cat=True, allow_qq=False, include_ecm=True)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    return parser

# Parser for a-FSR/plots.py
def fsr_plot_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'FSR recovery Plot Script',
        include_cat=True, allow_qq=False, include_ecm=True)
    add_procs_args(parser)
    add_plot_args(parser, False, False)
    return parser



#########################
### PARSER FOR b-CHI2 ###
#########################

# Parser for b-Chi2/pre-selection.py
def chi2_preselection_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Chi2 Optimization Pre-selection Script',
        include_cat=True, allow_qq=False, include_ecm=True)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    return parser

# Parser for b-Chi2/optimize.py
def chi2_optimization_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Chi2 Optimization Script',
        include_cat=True, allow_qq=False, include_ecm=True)
    add_procs_args(parser)
    add_optimize_args(parser)
    add_chi2_optimization_args(parser)
    return parser

# Parser for b-Chi2/plots.py
def chi2_plots_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(
        description or 'Chi2 Plot Script',
        include_cat=True, allow_qq=False, include_ecm=True)
    add_procs_args(parser)
    add_plot_args(parser, False, False)
    add_optimize_args(parser)
    add_plots_chi2_args(parser)
    return parser



########################
### PARSER FOR 0-RUN ###
########################

# Parser for 0-Run/a-run.py
def fsr_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run FSR recovery pipeline',
                         include_cat=True, cat_multi=True, cat_default='mumu', allow_qq=False,
                         include_ecm=True, ecm_multi=True, ecm_default='240')
    add_run_argument(parser, 2)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    add_procs_args(parser)
    return parser

# Parser for 0-Run/b-run.py
def chi2_parser(description: str | None = None) -> ArgumentParser:
    parser = base_parser(description or 'Run Chi2 Optimization pipeline',
                         include_cat=True, cat_multi=True, cat_default='mumu', allow_qq=False,
                         include_ecm=True, ecm_multi=True, ecm_default='240')
    add_run_argument(parser, 3)
    add_preselection_args(parser)
    add_sample_selection_args(parser)
    add_procs_args(parser)
    add_optimize_args(parser)
    add_chi2_optimization_args(parser)
    add_plots_chi2_args(parser)
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
        if directory == 'a-FSR':  return fsr_parser(description)
        if directory == 'b-Chi2': return chi2_parser(description)
        raise ValueError(f'Unknown parser directory: {directory}')

    # a-FSR directory
    if directory == 'a-FSR':
        if script == 'pre-selection': return fsr_preselection_parser(description)
        if script == 'plots':         return fsr_plot_parser(description)

    elif directory == 'b-Chi2':
        if script == 'pre-selection': return chi2_preselection_parser(description)
        if script == 'optimize':      return chi2_optimization_parser(description)
        if script == 'plots':         return chi2_plots_parser(description)

    raise ValueError(f'Unknown parser script: {directory}/{script}')
