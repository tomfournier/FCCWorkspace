################################
### STANDARD LIBRARY IMPORTS ###
################################

from time import time

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,        # Support multiple decay categories
    include_sels=True,     # Include selection strategy options
    cutflow=True,          # Include cutflow analysis options
    description='Cutflow Script'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory paths and cutflow analysis functions
from package.userConfig import loc
from package.config import (
    timer,              # Timing utility
    mk_processes,       # Build process definitions
    z_decays,           # Z boson decay modes
    H_decays,           # Higgs decay modes
    colors, labels      # Process styling for plots
)
from package.plots.cutflow import (
    get_cutflow,            # Calculate event counts per cut
    branches_from_cuts      # Get branches needed for each cut
)



############################
### GLOBAL CONFIGURATION ###
############################

# Decay categories to analyze (from command-line: cat1-cat2 format)
cats, ecm = arg.cat.split('-'), arg.ecm
# Integrated luminosity [ab^-1]
lumi = 10.8 if ecm==240 else (3.1 if ecm==365 else -1)

# Selection strategies to analyze (from command-line or defaults)
if arg.sels == '':
    sels = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'test']  # Default selections
else:
    sels = arg.sels.split('-')  # Parse from command-line



###################
### DEFINE CUTS ###
###################

# CoM-dependent kinematic bounds for baseline selection
p_up = 70 if ecm==240 else (150 if ecm==365 else 240)  # Upper momentum cut [GeV]
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)     # Lower momentum cut [GeV]

# Define sequential baseline selection cuts
baseline_cuts = {
    'cut0': '',                                                      # Diagnostic: no cuts
    'cut1': '',                                                      # Lepton isolation (applied in pre-selection)
    'cut2': '',                                                      # Opposite-sign requirement
    'cut3': '' if arg.kin else  'zll_m > 86 & zll_m < 96',           # Z mass window [GeV]
    'cut4': '' if arg.kin else f'zll_p > {p_dw} & zll_p < {p_up}'    # Dilepton momentum window
}

# Human-readable labels for cuts (displayed in plots)
baseline_labels = {
    'cut0': 'No cut',                                            # Diagnostic baseline
    'cut1': '#geq 1#ell^{#pm} + ISO',                            # Lepton with isolation
    'cut2': '#geq 2 #ell^{#pm} + OS',                            # Dilepton pair, opposite sign
    'cut3': '86 < m_{#ell^{+}#ell^{-}} < 96 GeV',                # Z boson mass selection
    'cut4': f'{p_dw} < p_{{#ell^{{+}}#ell^{{-}}}} < {p_up} GeV'  # Momentum selection
}
if ecm == 365:
    baseline_cuts['cut5']   = '' if arg.kin else 'zll_recoil_m > 100 && zll_recoil_m < 150'  # Recoil mass selection (if 365 GeV)
    baseline_labels['cut5'] = '100 < m_{recoil} < 150 GeV'

# Copy baseline cuts for each selection strategy
cuts       = {sel: baseline_cuts.copy()   for sel in sels}
cuts_label = {sel: baseline_labels.copy() for sel in sels}

# Add additional cuts for specific selection strategies
cuts['Baseline_miss']['cut5']       = 'cosTheta_miss < 0.98'
cuts_label['Baseline_miss']['cut5'] = 'cos#theta_{miss} < 0.98'

vis_cut = 100 if ecm==240 else (170 if ecm==365 else 0)
cuts['Baseline_sep']['cut5']       = f'((visibleEnergy > {vis_cut}) | (visibleEnergy < {vis_cut} & cosTheta_miss < 0.99))'
cuts_label['Baseline_sep']['cut5'] = 'cos#theta_{miss} < 0.99 [inv]'

cuts['test']['cut5']       = 'zll_recoil_m > 100 & zll_recoil_m < 150'
cuts_label['test']['cut5'] = '100 < m_{recoil} < 150 GeV'

# Variables required for cutflow evaluation (must match those used in cuts)
variables = [
    'leading_p',    'leading_theta',
    'subleading_p', 'subleading_theta',
    'zll_m', 'zll_p', 'zll_theta', 'zll_recoil_m',
    'acolinearity',  'acoplanarity',  'deltaR',
    'visibleEnergy', 'cosTheta_miss', 'missingMass',
    'H'
]



##########################
### EXECUTION FUNCTION ###
##########################

def run(cats: list[str],
        sels: list[str],
        processes: dict[str, list[str]],
        colors: dict[str, dict[str, str]],
        legend: dict[str, dict[str, str]]
        ) -> None:
    '''Generate cutflow plots for each channel and selection strategy.'''
    for cat in cats:
        LOGGER.info(f'Making plots for {cat} channel')
        # Define process names (signal must be first)
        procs = [f'Z{cat}H' if not arg.tot else 'ZH', 'WW', 'ZZ', 'Zgamma', 'Rare']
        procs_decays = [f'z{cat}h' if not arg.tot else 'zh', 'WW', 'ZZ', 'Zgamma', 'Rare']

        # Define input and output directories
        if arg.kin: inDir = loc.get('EVENTS_TEST', cat, ecm)
        else:       inDir = loc.get('EVENTS',      cat, ecm)
        outDir = loc.get('PLOTS_MEASUREMENT',      cat, ecm)

        # Extract required branches from cuts and variables
        branches = branches_from_cuts(cuts, variables)
        br_str = ', '.join(br for br in branches)
        LOGGER.info(f'Only importing these branches from the .root file: {br_str}')

        # Generate cutflow plots and tables
        get_cutflow(
            inDir, outDir,
            cat, sels,
            procs, procs_decays,
            processes, colors, legend,
            cuts, cuts_label,
            z_decays, H_decays,
            branches=branches,
            ecm=ecm,
            lumi=lumi,
            sig_scale=1,
            tot=arg.tot,
            loc_json=loc.JSON+f'/{cat}'
        )


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Run cutflow analysis for all categories and selections
        run(cats, sels, mk_processes(ecm=ecm), colors, labels)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
