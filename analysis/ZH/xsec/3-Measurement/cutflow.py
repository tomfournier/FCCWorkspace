##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from time import time
from argparse import ArgumentParser

# Start timer for performance tracking
t = time()

from package.userConfig import loc
from package.config import (
    timer, mk_processes,
    z_decays, H_decays, 
    colors, labels)
from package.plots.cutflow import (
    get_cutflow, 
    branches_from_cuts
)



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser()
# Define final state: ee, mumu, or both
parser.add_argument('--cat',  help='Final state (ee, mumu)', 
                    choices=['ee', 'mumu', 'ee-mumu'], type=str, default='ee-mumu')
# Define center of mass energy
parser.add_argument('--ecm',  help='Center of mass energy (240, 365)',
                    choices=[240, 365], type=int, default=240)

# Include all Z decay modes in plots
parser.add_argument('--tot', help='Include all the Z decays in the plots',
                    action='store_true')
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm = arg.cat.split('-'), arg.ecm
lumi = 10.8 if ecm==240 else (3.1 if ecm==365 else -1)
# Selection strategies to analyze
sels = [
    'Baseline', 'Baseline_miss', 'Baseline_sep', 'test'
]



#######################
### DEFINE THE CUTS ###
#######################

# Define Baseline selection cuts for each stage
p_up = 70 if ecm==240 else (150 if ecm==365 else 240)
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)
baseline_cuts = {
    'cut0': '', 'cut1': '', 'cut2': '',
    'cut3': 'zll_m > 86 & zll_m < 96',
    'cut4': f'zll_p > {p_dw} & zll_p < {p_up}'
}
# Human-readable labels for baseline cuts
baseline_labels = {
    'cut0': 'No cut',
    'cut1': '#geq 1#ell^{#pm} + ISO',
    'cut2': '#geq 2 #ell^{#pm} + OS',
    'cut3': '86 < m_{#ell^{+}#ell^{-}} < 96 GeV',
    'cut4': f'{p_dw} < p_{{#ell^{{+}}#ell^{{-}}}} < {p_up} GeV'
}

# Copy baseline cuts for each selection strategy
cuts       = {sel: baseline_cuts.copy()   for sel in sels}
cuts_label = {sel: baseline_labels.copy() for sel in sels}

# Add additional cuts for specific selection strategies
cuts['Baseline_miss']['cut5'] = 'cosTheta_miss < 0.98'
cuts_label['Baseline_miss']['cut5'] = 'cos#theta_{miss} < 0.98'

vis_cut = 100 if ecm==240 else (170 if ecm==365 else 0)
cuts['Baseline_sep']['cut5'] = f'((visibleEnergy > {vis_cut}) | (visibleEnergy < {vis_cut} & cosTheta_miss < 0.99))'
cuts_label['Baseline_sep']['cut5'] = 'cos#theta_{miss} < 0.99 [inv]'

cuts['test']['cut5'] = 'zll_recoil_m > 100 & zll_recoil_m < 150'
cuts_label['test']['cut5'] = '100 < m_{recoil} < 150 GeV'

# Variables required for cutflow evaluation (must match those used in cuts)
variables = [
    'leading_p',    'leading_theta', 
    'subleading_p', 'subleading_theta',
    'zll_m', 'zll_p', 'zll_theta', 'zll_recoil_m',
    'acolinearity',  'acoplanarity',  'zll_deltaR',
    'visibleEnergy', 'cosTheta_miss', 'missingMass',
    'H', 'BDTscore'
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
        print(f'\n----->[Info] Making plots for {cat} channel\n')
        # Define process names (signal must be first)
        procs = [f'Z{cat}H' if not arg.tot else 'ZH', 'WW', 'ZZ', 'Zgamma', 'Rare']
        procs_decays = [f'z{cat}h' if not arg.tot else 'zh', 'WW', 'ZZ', 'Zgamma', 'Rare']

        # Define input and output directories
        inDir  = loc.get('EVENTS', cat, ecm)
        outDir = loc.get('PLOTS_MEASUREMENT', cat, ecm)

        # Extract required branches from cuts and variables
        branches = branches_from_cuts(cuts, variables)
        br_str = ', '.join(br for br in branches)
        print(f'----->[Info] Only importing these branches from the .root file: \n\t{br_str}\n')

        # Generate cutflow plots and tables
        get_cutflow(inDir, outDir,
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
                    loc_json=loc.JSON+f'/{cat}')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run cutflow analysis for all categories and selections
    run(cats, sels, mk_processes(ecm=ecm), colors, labels)
    # Print execution time
    timer(t)
