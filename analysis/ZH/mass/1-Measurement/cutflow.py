#################################
### IMPORT STANDARD LIBRARIES ###
#################################

from time import time

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,
    include_sels=True,
    cutflow=True,
    description='Cutflow Script'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
from sel.final import histo_list
from package.config import (
    timer, mk_processes,
    colors, labels
)
from package.plots.cutflow import (
    get_cutflow,
    branches_from_cuts
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm = arg.cat.split('-'), arg.ecm
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)
# Selection strategies to analyze
if arg.sels == '':
    sels = ['Baseline']
else:
    sels = arg.sels.split('-')



#######################
### DEFINE THE CUTS ###
#######################

# Define Baseline selection cuts for each stage
p_up = 70 if ecm==240 else (150 if ecm==365 else 240)
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)
baseline_cuts = {
    'cut0': '', 'cut1': '', 'cut2': '',
    'cut3': 'zll_m > 86 & zll_m < 96',
}
# Human-readable labels for baseline cuts
baseline_labels = {
    'cut0': 'No cut',
    'cut1': '#geq 1#ell^{#pm} + ISO',
    'cut2': '#geq 2 #ell^{#pm} + OS',
    'cut3': '86 < m_{#ell^{+}#ell^{-}} < 96 GeV',
}
if ecm == 240:
    baseline_cuts['cut4']   = 'zll_p > 20 & zll_p < 70'
    baseline_labels['cut4'] = '20 < p_{#ell^{+}#ell^{-}} < 70 GeV'

    baseline_cuts['cut5']   = 'cosTheta_miss < 0.98'
    baseline_labels['cut5'] = 'cos#theta_{miss} < 0.98'
elif ecm == 365:
    baseline_cuts['cut4']   = 'zll_p > 50 & zll_p < 150'
    baseline_labels['cut4'] = '50 < p_{#ell^{+}#ell^{-}} < 150 GeV'

    baseline_cuts['cut5']   = 'zll_recoil_m > 100 & zll_recoil_m < 150'
    baseline_labels['cut5'] = '100 < m_{recoil} < 150 GeV'

    baseline_cuts['cut6']   = 'cosTheta_miss < 0.98'
    baseline_labels['cut6'] = 'cos#theta_{miss} < 0.98'

# Copy baseline cuts for each selection strategy
cuts       = {sel: baseline_cuts.copy()   for sel in sels}
cuts_label = {sel: baseline_labels.copy() for sel in sels}

# Variables required for cutflow evaluation (must match those used in cuts)
variables = histo_list.keys()



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
        inDir  = loc.get('EVENTS_TEST', cat, ecm)
        outDir = loc.get('PLOTS_MEASUREMENT', cat, ecm)

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
