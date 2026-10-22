################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging



########################
### ARGUMENT PARSING ###
########################

from zh_xsec.parsing import create_parser
parser = create_parser('3-Measurement', 'final-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.final-selection')



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_xsec.userConfig import loc
from zh_xsec.func.bdt   import def_bdt, make_high_low
from zh_xsec.sel.final.leptonic import Baseline_cut_ll, histos_ll, custom_hists_ll
from zh_xsec.sel.final.hadronic import Baseline_cut_qq, histos_qq, custom_hists_qq
from samples import get_process_list, get_include, get_exclude
from utilities import has_events



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

# Input: Preprocessed ROOT trees and events from pre-selection
if   test: inputDir = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
elif jan:  inputDir = loc.get('EVENTS_JAN',  cat, ecm)  # Jan's samples
else:      inputDir = loc.get('EVENTS',      cat, ecm)  # Full event samples



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
# saveJSON    = True  # Export results to JSON format
# saveTabular = True  # Generate LaTeX tables



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Samples to process: ZH signal and main background processes
# These are processed through final selection cuts and histogram filling
samples = get_process_list(
    cat, ecm, onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=get_include(arg.include),
    exclude=get_exclude(arg.exclude)
).keys()

# Load event samples with events TTree
processList = has_events(list(samples), inputDir)

# Define BDT score from trained model and apply BDT cut
if test: loc_BDT = loc.get('MVA_MODEL', cat, ecm, arg.bdt_sel if arg.bdt_sel else 'test')
else:    loc_BDT = loc.get('MVA_MODEL', cat, ecm, arg.bdt_sel if arg.bdt_sel else 'Baseline')
LOGGER.info(f'Using BDT from {loc_BDT}')
defineList, bdt_cut = def_bdt(loc_BDT, weight_suffix=f'_{arg.weight_suffix}')



#######################
### DEFINE CUT LIST ###
#######################

# Baseline and Baseline_miss definition (channel and c.o.m energy dependent)
Baseline      = Baseline_cut_qq(ecm, False) if cat=='qq' else Baseline_cut_ll(ecm, False)
Baseline_miss = Baseline_cut_qq(ecm, True)  if cat=='qq' else Baseline_cut_ll(ecm, True)

# E_vis cut for E_vis categorization (channel and c.o.m energy dependent)
E_vis = (120 if ecm==240 else 175) if cat=='qq' else (100 if ecm==240 else 171)
# cos(theta_miss) cut for E_vis categorization (channel dependent)
theta_miss = 0.995 if cat=='qq' else 0.99


# Selection cut dictionary (key = selection name used in outputs)
cutList: dict[str, str] = {
    'Baseline':      Baseline,
    'Baseline_miss': Baseline_miss,
    'Baseline_sep':  Baseline + f' && ((visibleEnergy > {E_vis}) || (visibleEnergy < {E_vis} && cosTheta_miss < {theta_miss}))',
    'Baseline_vis':  Baseline + f' &&   visibleEnergy > {E_vis}',
    'Baseline_inv':  Baseline + f' &&   visibleEnergy < {E_vis}'
}
if arg.do_sel0:
    sel0 = 'sel0' + ('_test' if test else ('_jan' if jan else ''))
    cutList[sel0] = 'return true;'    # No cuts selection (optional)
if  test: cutList['test'] = Baseline  # Selection for testing
elif jan: cutList['jan']  = Baseline  # Selection for Jan's jet definition

# Select selection to process and split them into high/low BDT score regions
cutList = {sel:cuts for sel, cuts in cutList.items() if (sel in sels or 'all' in sels)}
cutList = make_high_low(cutList, bdt_cut, list(cutList.keys()))



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Custom histogram made at the pre-selection
customHists = custom_hists_ll if cat in ['ee', 'mumu'] else custom_hists_qq

# Output histogram definitions (name, title, binning)
histoList = histos_ll if cat in ['ee', 'mumu'] else histos_qq
histoList['BDTscore'] = {'name':'BDTscore', 'title':'BDT score', 'bin':1000,'xmin':0,'xmax':1}
if cat == 'qq':
    mqq_lim  = (100, 40, 140) if ecm==240 else (140, 60, 200)
    mrec_lim = (140, 60, 200) if ecm==240 else (215, 50, 265)
    bdt_low, bdt_high = (1, 0, bdt_cut), (1, bdt_cut, 1)
    histoList['zqq_m_recoil_m_mva_high']      = {'cols':['BDTscore', 'zqq_recoil_m', 'zqq_m'], 'bins':[bdt_high, mrec_lim, mqq_lim]}
    histoList['zqq_m_recoil_m_mva_low']       = {'cols':['BDTscore', 'zqq_recoil_m', 'zqq_m'], 'bins':[bdt_low,  mrec_lim, mqq_lim]}
    histoList['zqq_m_recoil_m_tot_mva_high']  = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(150, 50, 200), (100, 40, 140), (1, bdt_cut, 1)]}
    histoList['zqq_m_recoil_m_tot_mva_low']   = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(150, 50, 200), (100, 40, 140), (1, 0, bdt_cut)]}
