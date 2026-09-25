################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser
parser = create_parser('1-MVAInputs', 'final-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.final-selection')



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load analysis configuration and predefined histogram config
from package.userConfig import loc
from package.config import (
    get_process_list,
    parse_sample_selection,
    parse_sample_exclusion
)
from sel.final.leptonic import Baseline_cut_ll, histos_ll
from sel.final.hadronic import Baseline_cut_qq, histos_qq



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

cat, ecm, sels, test = arg.cat, arg.ecm, arg.sels.split('-'), arg.test
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)
if test and arg.jan: raise ValueError("--test and --jan can't be used together, choose one")
LOGGER.info(f'Using {cat = } | {ecm = } | sels = {arg.sels} | {test = } | jan = {arg.jan}')

# Input: Pre-selection ROOT trees and histograms
if test:      inputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)  # Test subset
elif arg.jan: inputDir = loc.get('EVENTS_TRAIN_JAN',  cat, ecm)  # Jan's samples
else:         inputDir = loc.get('EVENTS_TRAINING',   cat, ecm)  # Full training sample

# Output: Directory for MVA input histograms
outputDir = loc.get('HIST_MVA', cat, ecm)

# Process dictionary with cross-section and sample metadata
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

# Parallel processing configuration (default 4)
nCPUS = 10  # Number of CPUs for parallel histogram filling

# ROOT output options
doScale = True        # Scale histograms to integrated luminosity
intLumi = lumi * 1e6  # Integrated luminosity in pb^-1

# Optional outputs (commented out by default)
# saveJSON = True    # Export results to JSON format
# saveTabular = True # Generate LaTeX tables



##########################
### DEFINE SAMPLE LIST ###
##########################

# Samples to process: ZH signal and main background processes
# These are processed through final selection cuts and histogram filling
processList = get_process_list(
    cat, ecm, train=True,
    onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=parse_sample_selection(arg.include),
    exclude=parse_sample_exclusion(arg.exclude)
).keys()



########################
### DEFINE SELECTION ###
########################

# Selection cuts dictionary for ROOT filtering
# Keys: selection names appearing in output file names and histograms
cutList: dict[str, str] = {}
if arg.do_sel0: cutList['sel0_test' if arg.test else 'sel0'] = 'return true;'  # No cuts selection
if cat in ['ee', 'mumu']:
    if test: cutList['test']     = Baseline_cut_ll(ecm)   # Test selection (leptonic channel)
    else:    cutList['Baseline'] = Baseline_cut_ll(ecm)   # Baseline selection (leptonic channel)
elif cat == 'qq':
    if test: cutList['test']     = Baseline_cut_qq(ecm, True) + ' && delta_mWW4 > 6'   # Test selection     (hadronic channel)
    if arg.jan:
        cutList['jan']  = Baseline_cut_qq(ecm, True)
        cutList['jan1'] = Baseline_cut_qq(ecm, True) + ' && delta_mWW4 > 6'
        cutList['jan2'] = Baseline_cut_qq(ecm, True) + ' && delta_mWW4 > 6 && acolinearity > 0.35'
        cutList['jan3'] = Baseline_cut_qq(ecm, True) + ' && delta_mWW4 > 6 && zqq_costheta < 0.85 && zqq_costheta > -0.85'
        cutList['jan4'] = Baseline_cut_qq(ecm, True) + ' && delta_mWW4 > 6 && acolinearity > 0.35 && zqq_costheta < 0.85 && zqq_costheta > -0.85'
    else:    cutList['Baseline'] = Baseline_cut_qq(ecm, True) + ' && delta_mWW4 > 6'   # Baseline selection (hadronic channel)
cutList = {sel:cuts for sel, cuts in cutList.items() if (sel in sels or 'all' in sels)}

# Save ROOT TTrees in addition to histograms (for BDT training)
doTree = False if 'sel0' in cutList else arg.do_tree  # Do not write TTree if sel0 is in cutList



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Output histogram definitions (name, title, binning)
if cat in ['ee', 'mumu']: histoList = histos_ll
elif cat == 'qq':         histoList = histos_qq
else: raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
