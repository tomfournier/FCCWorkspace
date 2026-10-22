################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging



########################
### ARGUMENT PARSING ###
########################

from zh_xsec.parsing import create_parser
parser = create_parser('1-MVAInputs', 'final-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.final-selection')



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load analysis configuration and predefined histogram config
from zh_xsec.userConfig import loc
from samples import get_process_list, get_include, get_exclude
from zh_xsec.sel.final.leptonic import Baseline_cut_ll, histos_ll
from zh_xsec.sel.final.hadronic import Baseline_cut_qq, histos_qq



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

cat, ecm, sels, test, jan = arg.cat, arg.ecm, arg.sels.split('-'), arg.do_test, arg.jan
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)
if jan and test:
    raise ValueError('Cannot use --do-test and --jan at the same time, Choose one at a time')
if jan and cat in ['ee', 'mumu']:
    raise ValueError("Cannot use --jan with cat = 'ee', 'mumu'. Use it with cat = 'qq'")
LOGGER.info(f'Running the final-selection for {cat = } | {ecm = } | {test = } | {jan = }')

# Input: Pre-selection ROOT trees and histograms
if   test:    inputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)  # Test subset
elif arg.jan: inputDir = loc.get('EVENTS_TRAIN_JAN',  cat, ecm)  # Jan's samples
else:         inputDir = loc.get('EVENTS_TRAIN',      cat, ecm)  # Full training sample

# Output: Directory for MVA input histograms
outputDir = loc.get('HIST_INPUT', cat, ecm)

# Process dictionary with cross-section and sample metadata
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

# Parallel processing configuration (default 4)
nCPUS = 10  # Number of CPUs for parallel histogram filling

# ROOT output options
doScale = True        # Scale histograms to integrated luminosity
intLumi = lumi * 1e6  # Integrated luminosity in pb^-1

# Optional outputs (commented out by default)
# saveJSON    = True  # Export results to JSON format
# saveTabular = True  # Generate LaTeX tables



##########################
### DEFINE SAMPLE LIST ###
##########################

# Samples to process: ZH signal and main background processes
# These are processed through final selection cuts and histogram filling
processList = get_process_list(
    cat, ecm, train=True,
    onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=get_include(arg.include),
    exclude=get_exclude(arg.exclude)
).keys()



########################
### DEFINE SELECTION ###
########################

Baseline = Baseline_cut_qq(ecm, True) if cat=='qq' else Baseline_cut_ll(ecm)

# Selection cuts dictionary for ROOT filtering
# Keys: selection names appearing in output file names and histograms
cutList: dict[str, str] = {}
if arg.do_sel0:
    sel0 = 'sel0' + ('_test' if test else ('_jan' if jan else ''))
    cutList[sel0] = 'return true;'        # No cuts selection (optional)
if  test: cutList['test']     = Baseline  # Selection for testing
elif jan: cutList['jan']      = Baseline  # Selection for Jan's jet definition
else:     cutList['Baseline'] = Baseline  # Selection for Baseline

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
