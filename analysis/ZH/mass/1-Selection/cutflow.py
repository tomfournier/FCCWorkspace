#################################
### IMPORT STANDARD LIBRARIES ###
#################################

from time import time

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_mass.parsing import create_parser
from logger import get_logger
arg = create_parser('1-Selection', 'cutflow').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_mass.userConfig import loc
from tools.utils import timer
from samples import get_process_dict_mass
from plotting.cutflow import get_cutflow
from constants import colors, legend



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Decay categories to analyze (from command-line: cat1-cat2 format)
cat, ecm, sels = arg.cat, arg.ecm, arg.sels.split('-')
# Integrated luminosity [ab^-1]
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

# Define input and output directories
if arg.kin: inputdir = loc.get('EVENTS_TEST', cat, ecm)
else:       inputdir = loc.get('EVENTS',      cat, ecm)
outputdir = loc.get('PLOT_MEAS', cat, ecm)



#######################
### DEFINE THE CUTS ###
#######################

CUTS = {
    'cut0': ('', 'No cut'),                # No cut
    'cut1': ('', '#geq 1 l^{#pm} + ISO'),  # At least one isolated lepton
    'cut2': ('', '#geq 2 l^{#pm} + OS'),   # Di-lepton pair, opposite sign
    'cut3': ('zll_m > 86 & zll_m < 96', '86 < m_{l^{+}l^{-}} < 96 GeV'),                   # Z boson mass selection
    'cut4': ('zll_p > 20 & zll_p < 70',  '20 < p_{l^{+}l^{-}} < 70 GeV') if ecm==240 else  # Di-lepton momentum window (240 GeV)
            ('zll_p > 50 & zll_p < 150', '50 < p_{l^{+}l^{-}} < 150 GeV'),                 # Di-lepton momentum window (365 GeV)
    **({'cut5': (('zll_recoil_m > 100 & zll_recoil_m < 150',
                  '100 < m_{recoil} < 150 GeV'))} if ecm==365 else {})
}

if arg.kin:
    remove = {'cut3', 'cut4'} + {'cut5'} if ecm==365 else set()
    CUTS = {k: ('' if k in remove else cut, label) for k, (cut, label) in CUTS.items()}

new_cut = 'cut5' if ecm==240 else 'cut6'
E_vis = 100 if ecm==240 else 171
EXTRA = {'Baseline_miss': {new_cut: ('cosTheta_miss < 0.98', '|cos#theta_{miss}| < 0.98')},  # Missing energy polar angle cut
         'Baseline_sep':  {new_cut: (f'((visibleEnergy > {E_vis}) |'                         # Visible sample, do nothing
                                     f' (visibleEnergy < {E_vis} & cosTheta_miss < 0.99))',  # Invisible sample, missing energy polar angle cut
                                      '|cos#theta_{miss}| < 0.99 [inv]')}}

# Merge baseline and selection-specific cuts before splitting expressions and labels.
CUTS_BY_SEL = {sel: {**CUTS, **EXTRA.get(sel, {})} for sel in sels}

cuts   = {sel: {k: specs[k][0] for k in sorted(specs.keys())} for sel, specs in CUTS_BY_SEL.items()}
labels = {sel: {k: specs[k][1] for k in sorted(specs.keys())} for sel, specs in CUTS_BY_SEL.items()}



####################################
### PROCESSES DICT CONFIGURATION ###
####################################

plots = {'signals': get_process_dict_mass(['ZH', f'Z{cat}H'], ecm),
         'backgrounds': get_process_dict_mass(['WW', 'ZZ', 'Zgamma', 'Rare'], ecm)}



##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    '''Generate cutflow plots for each selection strategy.'''
    LOGGER.info(f'Making plots for {cat} channel')

    # Generate cutflow plots and tables
    # TO DO: Adapt get_cutflow to cutflow mass
    get_cutflow(inputdir, outputdir, cat, sels, plots, colors, legend,
                cuts, labels, arg.formats.split('-'), ecm, lumi,
                1, True, arg.tot, True, loc.JSON)


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
