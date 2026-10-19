################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging



########################
### ARGUMENT PARSING ###
########################

from zh_mass.parsing import create_parser
parser = create_parser('1-Selection', 'final-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.final-selection')



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_mass.userConfig import loc
from zh_mass.sel.final import Baseline_cut, histo_list
from samples import get_process_list, get_include, get_exclude
from utilities import has_events



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

cat, ecm, sels, test = arg.cat, arg.ecm ,arg.sels.split('-'), arg.do_test
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

# Input: Preprocessed ROOT trees and events from pre-selection
if test: inputDir  = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
else:    inputDir  = loc.get('EVENTS',      cat, ecm)  # Full event sample

# Output: Directory for measurement histograms (used by measurement/fit stages)
outputDir = loc.get('HIST_RAW', cat, ecm)

# Process dictionary with cross-section and sample metadata
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# Parallel processing configuration
nCPUS = 10  # Number of CPUs for parallel histogram filling

# ROOT output options
doScale = True        # Scale histograms to integrated luminosity
intLumi = lumi * 1e6  # Integrated luminosity in pb^-1

# Optional outputs (commented out by default)
# saveJSON = True    # Export results to JSON format
# saveTabular = True # Generate LaTeX tables



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Samples processed during pre-selection
samples = get_process_list(
    cat, ecm, onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=get_include(arg.include),
    exclude=get_exclude(arg.exclude)
).keys()

# Load event samples with events TTree
processList = has_events(samples, inputDir)



#######################
### DEFINE CUT LIST ###
#######################

Baseline = Baseline_cut(ecm)
E_vis = 100 if ecm==240 else (171 if ecm==365 else -1)

# Selection cut dictionary (key = selection name used in outputs)
cutList: dict[str, str] = {
    'Baseline':      Baseline,
    'Baseline_miss': Baseline +  ' && cosTheta_miss < 0.98',
    'Baseline_vis':  Baseline + f' &&   visibleEnergy > {E_vis}',
    'Baseline_inv':  Baseline + f' &&   visibleEnergy < {E_vis}',
    'Baseline_sep':  Baseline + f' && ((visibleEnergy > {E_vis}) || '
                                     f'(visibleEnergy < {E_vis} && cosTheta_miss < 0.99))'
}
if arg.do_sel0: cutList['sel0_test' if test else 'sel0'] = 'return true;'
cutList = {sel:cuts for sel, cuts in cutList.items() if (sel in sels or 'all' in sels)}



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

customHists: dict[str, dict[str, str | int | float]] = {}
if cat in ['ee', 'mumu']:
    customHists['leps_iso']    = {'name':'ConeIsolation', 'title':'I_{rel}'}
    customHists['leps_iso_no'] = {'name':'n_leptons',     'title':'Isolated leptons'}

histoList = histo_list
