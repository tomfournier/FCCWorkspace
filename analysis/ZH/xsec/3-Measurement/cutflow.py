################################
### STANDARD LIBRARY IMPORTS ###
################################

from time import time

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_xsec.parsing import create_parser
from logger import get_logger
parser = create_parser('3-Measurement', 'cutflow')
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory paths and cutflow analysis functions
from zh_xsec.userConfig import loc
from tools.utils import timer
from samples import get_process_dict
from constants import QUARKS, z_decays, H_decays, colors, legend
from plotting.cutflow import get_cutflow



############################
### GLOBAL CONFIGURATION ###
############################

# Decay categories to analyze (from command-line: cat1-cat2 format)
cat, ecm, sels = arg.cat, arg.ecm, arg.sels.split('-')
# Integrated luminosity [ab^-1]
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)



###################
### DEFINE CUTS ###
###################

# Leptonic cuts and labels
CUTS_LL = {
    'cut0': ('', 'No cut'),                # No cut
    'cut1': ('', '#geq 1 l^{#pm} + ISO'),  # At least one isolated lepton
    'cut2': ('', '#geq 2 l^{#pm} + OS'),   # Di-lepton pair, opposite sign
    'cut3': ('zll_m > 86 & zll_m < 96', '86 < m_{l^{+}l^{-}} < 96 GeV'),                   # Z boson mass selection
    'cut4': ('zll_p > 20 & zll_p < 70',  '20 < p_{l^{+}l^{-}} < 70 GeV') if ecm==240 else  # Di-lepton momentum window (240 GeV)
            ('zll_p > 50 & zll_p < 150', '50 < p_{l^{+}l^{-}} < 150 GeV'),                 # Di-lepton momentum window (365 GeV)
    **({'cut5': (('zll_recoil_m > 100 & zll_recoil_m < 150',
                  '100 < m_{recoil} < 150 GeV'))} if ecm==365 else {})
}

if arg.kin: CUTS_LL = {k: ('', label) for k, (_, label) in CUTS_LL.items()}

new_cut = 'cut5' if ecm==240 else 'cut6'
E_vis = 100 if ecm==240 else 171
EXTRA_LL = {'Baseline_miss': {new_cut: ('cosTheta_miss < 0.98', '|cos#theta_{miss}| < 0.98')},  # Missing energy polar angle cut
            'Baseline_sep':  {new_cut: (f'((visibleEnergy > {E_vis}) |'                         # Visible sample, do nothing
                                        f' (visibleEnergy < {E_vis} & cosTheta_miss < 0.99))',  # Invisible sample, missing energy polar angle cut
                                        '|cos#theta_{miss}| < 0.99 [inv]')}}

# Hadronic cuts and labels
CUTS_QQ = {
    'cut0': ('', 'No cut'),         # No cut
    'cut1': ('', 'Veto leptonic'),  # Remove events passing the leptonic selection (orthogonality)
    'cut2': ('', 'Clustering'),     # Remove events failing the clustering
    'cut3': ('zqq_m > 20 & zqq_m < 140', '20 < m_{jj} < 140 GeV') if ecm==240 else      # Z boson mass selection (240 GeV)
            ('zqq_m > 60 & zqq_m < 200', '60 < m_{jj} < 200 GeV'),                      # Z boson mass selection (240 GeV)
    'cut4': ('zqq_p > 20 & zqq_p < 90',  '20 < p_{jj} < 90 GeV') if ecm==240 else       # Di-jet momentum window (240 GeV)
            ('zqq_p > 20 & zqq_p < 160', '20 < p_{jj} < 160 GeV'),                      # Di-jet momentum window (240 GeV)
    'cut5': ('zqq_costheta > -0.85 & zqq_costheta < 0.85', '|cos#theta_{jj}| < 0.85'),  # Di-jet polar angle cut
    'cut6': ('delta_mWW4 > 6', 'WW pair mass'),                      # Remove WW-like events
    'cut7': ('cosTheta_miss < 0.995', '|cos#theta_{miss}| < 0.995')  # Missing energy polar angle cut
}

if arg.kin:
    CUTS_QQ = {k: ('' if k in {'cut3', 'cut4'} else cut, label) for k, (cut, label) in CUTS_QQ.items()}

E_vis = 120 if ecm==240 else 175
EXTRA_QQ = {'Baseline_sep':  {new_cut: (f'((visibleEnergy > {E_vis}) |'                          # Visible sample, do nothing
                                        f' (visibleEnergy < {E_vis} & cosTheta_miss < 0.995))',  # Invisible sample, missing energy polar angle cut
                                        '|cos#theta_{miss}| < 0.995 [inv]')}}



################################
### PROCESS CUTS FOR CUTFLOW ###
################################

# Merge baseline and selection-specific cuts before splitting expressions and labels.
CUTS_LL_BY_SEL = {sel: {**CUTS_LL, **EXTRA_LL.get(sel, {})} for sel in sels}

cuts_ll  = {sel: {k: specs[k][0] for k in sorted(specs.keys())} for sel, specs in CUTS_LL_BY_SEL.items()}
label_ll = {sel: {k: specs[k][1] for k in sorted(specs.keys())} for sel, specs in CUTS_LL_BY_SEL.items()}

# Merge baseline and selection-specific cuts before splitting expressions and labels.
CUTS_QQ_BY_SEL = {sel: {**CUTS_QQ, **EXTRA_QQ.get(sel, {})} for sel in sels}

cuts_qq  = {sel: {k: specs[k][0] for k in sorted(specs.keys())} for sel, specs in CUTS_QQ_BY_SEL.items()}
label_qq = {sel: {k: specs[k][1] for k in sorted(specs.keys())} for sel, specs in CUTS_QQ_BY_SEL.items()}



####################################
### PROCESSES DICT CONFIGURATION ###
####################################

decays_cat = [cat] if cat in ['ee', 'mumu'] else (QUARKS if cat=='qq' else [])
plots = {'signals':     get_process_dict(['ZH', f'Z{cat}H'], ecm),
         'backgrounds': get_process_dict(['WW', 'ZZ', 'Zgamma', 'Rare'], ecm),
         'decays': {'cat': {y: [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in decays_cat] for y in H_decays},
                    'tot': {y: [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays]   for y in H_decays}}}



##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    '''Generate cutflow plots for each channel and selection strategy.'''
    LOGGER.info(f'Making plots for {cat} channel')
    # Define process names (signal must be first)

    cuts   = cuts_ll  if cat in ['ee', 'mumu'] else cuts_qq
    labels = label_ll if cat in ['ee', 'mumu'] else label_qq

    # Define input and output directories
    if arg.kin: inDir = loc.get('EVENTS_TEST', cat, ecm)
    else:       inDir = loc.get('EVENTS',      cat, ecm)
    outDir = loc.get('PLOT_MEAS', cat, ecm)

    # Generate cutflow plots and tables
    get_cutflow(inDir, outDir, cat, sels, plots, colors, legend,
                cuts, labels, arg.formats.split('-'), ecm, lumi,
                1, arg.tot, True, True,
                loc.JSON)


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Run cutflow analysis for all categories and selections
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
