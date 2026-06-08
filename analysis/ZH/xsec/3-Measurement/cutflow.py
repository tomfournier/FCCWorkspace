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
from sel.final.leptonic import histos_ll
from sel.final.hadronic import histos_qq
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
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

# Selection strategies to analyze (from command-line or defaults)
if arg.sels == '':
    sels = ['Baseline']  # Default selection
else:
    sels = arg.sels.split('-')  # Parse from command-line



############################
### DEFINE LEPTONIC CUTS ###
############################

# Define sequential baseline selection cuts
baseline_cuts_ll = {
    'cut0': '',                                                      # Diagnostic: no cuts
    'cut1': '',                                                      # Lepton isolation (applied in pre-selection)
    'cut2': '',                                                      # Opposite-sign requirement
    'cut3': '' if arg.kin else  'zll_m > 86 & zll_m < 96',           # Z mass window [GeV]
}

# Human-readable labels for cuts (displayed in plots)
baseline_labels_ll = {
    'cut0': 'No cut',                                            # Diagnostic baseline
    'cut1': '#geq 1#ell^{#pm} + ISO',                            # Lepton with isolation
    'cut2': '#geq 2 #ell^{#pm} + OS',                            # Dilepton pair, opposite sign
    'cut3': '86 < m_{#ell^{+}#ell^{-}} < 96 GeV',                # Z boson mass selection
}

if ecm == 240:
    baseline_cuts_ll['cut4']   = '' if arg.kin else 'zll_p > 20 & zll_p < 70'                   # Dilepton momentum window (240 GeV)
    baseline_labels_ll['cut4'] = '20 < p_{#ell^{+}#ell^{-}} < 70 GeV'
elif ecm == 365:
    baseline_cuts_ll['cut4']   = '' if arg.kin else 'zll_p > 50 & zll_p < 150'                  # Dilepton momentum window (365 GeV)
    baseline_labels_ll['cut4'] = '50 < p_{#ell^{+}#ell^{-}} < 150 GeV'

    baseline_cuts_ll['cut5']   = '' if arg.kin else 'zll_recoil_m > 100 && zll_recoil_m < 150'  # Recoil mass selection (if 365 GeV)
    baseline_labels_ll['cut5'] = '100 < m_{recoil} < 150 GeV'


# Copy baseline cuts for each selection strategy
cuts_ll       = {sel: baseline_cuts_ll.copy()   for sel in sels}
cuts_label_ll = {sel: baseline_labels_ll.copy() for sel in sels}

# Add additional cuts for specific selection strategies
if ecm == 240:
    if 'Baseline_miss' in sels:
        cuts_ll['Baseline_miss']['cut5']       = 'cosTheta_miss < 0.98'
        cuts_label_ll['Baseline_miss']['cut5'] = 'cos#theta_{miss} < 0.98'

    if 'Baseline_sep' in sels:
        cuts_ll['Baseline_sep']['cut5']       = '((visibleEnergy > 100) | (visibleEnergy < 100 & cosTheta_miss < 0.99))'
        cuts_label_ll['Baseline_sep']['cut5'] = 'cos#theta_{miss} < 0.99 [inv]'
elif ecm == 365:
    if 'Baseline_miss' in sels:
        cuts_ll['Baseline_miss']['cut6']       = 'cosTheta_miss < 0.98'
        cuts_label_ll['Baseline_miss']['cut6'] = 'cos#theta_{miss} < 0.98'

    if 'Baseline_sep' in sels:
        cuts_ll['Baseline_sep']['cut6']       = '((visibleEnergy > 171) | (visibleEnergy < 171 & cosTheta_miss < 0.99))'
        cuts_label_ll['Baseline_sep']['cut6'] = 'cos#theta_{miss} < 0.99 [inv]'



###############################
### DEFINE HADRONICNIC CUTS ###
###############################

# Define sequential baseline selection cuts
baseline_cuts_qq = {
    'cut0': '',
    'cut1': '',
    'cut2': '',
    'cut5': '' if arg.kin else 'zqq_costheta < 0.85',
    'cut6': '' if arg.kin else 'acolinearity > 0.35',
    'cut7': '' if arg.kin else 'delta_mWW > 6',
    'cut8': '' if arg.kin else 'cosTheta_miss < 0.995',
}

# Human-readable labels for cuts (displayed in plots)
baseline_labels_qq = {
    'cut0': 'No cut',
    'cut1': 'Veto leptonic',
    'cut2': 'Clustering',
    'cut5': 'cos#theta_{jj} < 0.85',
    'cut6': 'Acolinearity > 0.35',
    'cut7': 'WW pair mass',
    'cut8': 'cos#theta_{miss} < 0.995',
}

if ecm == 240:
    baseline_cuts_qq['cut3']   = '' if arg.kin else 'zqq_m > 20 & zqq_m < 140'
    baseline_labels_qq['cut3'] = '20 < m_{jj} < 140'

    baseline_cuts_qq['cut4']   = '' if arg.kin else 'zqq_p > 20 & zqq_p < 90'
    baseline_labels_qq['cut4'] = '20 < p_{jj} < 70 GeV'

elif ecm == 365:
    baseline_cuts_qq['cut3']   = '' if arg.kin else 'zqq_m > 60 & zqq_m < 200'
    baseline_labels_qq['cut3'] = '60 < m_{jj} < 200'

    baseline_cuts_qq['cut4']   = '' if arg.kin else 'zqq_p > 20 & zqq_p < 160'
    baseline_labels_qq['cut4'] = '20 < p_{jj} < 160 GeV'

    baseline_cuts_qq['cut9']   = '' if arg.kin else 'thrust < 0.85'
    baseline_labels_qq['cut9'] = 'Thrust < 0.85'


# Copy baseline cuts for each selection strategy
cuts_qq       = {sel: baseline_cuts_qq.copy()   for sel in sels}
cuts_label_qq = {sel: baseline_labels_qq.copy() for sel in sels}



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

        histos = histos_ll if cat in ['ee', 'mumu'] else histos_qq
        variables = [v for v in histos.keys() if v!='zqq_m_recoil_m']

        cuts = cuts_ll if cat in ['ee', 'mumu'] else cuts_qq
        cuts_label = cuts_label_ll if cat in ['ee', 'mumu'] else cuts_label_qq

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
