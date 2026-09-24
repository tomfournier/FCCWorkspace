################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser
parser = create_parser(
    cat_single=True,
    include_sels=True,
    presel=True,
    is_final=True,
    description='Final-selection Script'
)
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknwon'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.final-selection')



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load configuration and measurement selection functions
from package.userConfig import loc, event
from package.func.bdt   import def_bdt, make_high_low  # BDT score binning utilities
from sel.final.leptonic import (
    Baseline_cut_ll,  # Baseline cut definition      (leptonic channel)
    histos_ll,        # Histogram definitions        (leptonic channel)
    custom_hists_ll   # Custom histogram definitions (leptonic channel)
)
from sel.final.hadronic import (
    Baseline_cut_qq,  # Baseline cut definition      (hadronic channel)
    histos_qq,        # Histogram definitions        (hadronic channel)
    custom_hists_qq   # Custom histogram definitions (hadronic channel)
)
from package.config import (
    get_process_list,
    parse_sample_selection,
    parse_sample_exclusion
)



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

cat, ecm, sels, test = arg.cat, arg.ecm, arg.sels.split('-'), arg.test
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

# Input: Preprocessed ROOT trees and events from pre-selection
if test: inputDir  = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
else:    inputDir  = loc.get('EVENTS',      cat, ecm)  # Full event sample

# Output: Directory for measurement histograms (used by measurement/fit stages)
outputDir = loc.get('HIST_PREPROCESSED', cat, ecm)

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

# Samples to process: ZH signal and main background processes
# These are processed through final selection cuts and histogram filling
samples = get_process_list(
    cat, ecm,
    onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=parse_sample_selection(arg.include),
    exclude=parse_sample_exclusion(arg.exclude)
).keys()

# Load event samples with events TTree
processList = event(samples, inputDir)

# Define BDT score from trained model and apply BDT cut
if test: loc_BDT = loc.get('BDT', cat, ecm, arg.bdt_sel if arg.bdt_sel else 'test')
else:    loc_BDT = loc.get('BDT', cat, ecm, arg.bdt_sel if arg.bdt_sel else 'Baseline')
defineList, bdt_cut = def_bdt(loc_BDT, weight_suffix=f'_{arg.weight_suffix}')



#######################
### DEFINE CUT LIST ###
#######################

Baseline = Baseline_cut_ll(ecm) if cat in ['ee', 'mumu'] else Baseline_cut_qq(ecm)

# Selection cut dictionary (key = selection name used in outputs)
cutList: dict[str, str] = {}
if arg.do_sel0: cutList['sel0_test' if arg.test else 'sel0'] = 'return true;'  # No cuts selection

if cat in ['ee', 'mumu']:
    if test:
        cutList['test'] = Baseline
    else:
        Baseline_miss = Baseline + ' && cosTheta_miss < 0.98'
        E_vis, theta_miss = 100 if ecm == 240 else 171, 0.99
elif cat == 'qq':
    Baseline_miss = Baseline_cut_qq(ecm, True)
    Baseline_old  = Baseline_miss + ' && acolinearity > 0.35 && zqq_costheta < 0.85 && zqq_costheta > -0.85'
    if test:
        cutList['test']   = Baseline_miss
        cutList['test1']  = Baseline_old
        cutList['test2']  = Baseline_old + ' && delta_mWW4 > 6'
        cutList['test3']  = Baseline_old + ' && delta_mWW4 > 9'
    else:
        E_vis, theta_miss = 120 if ecm == 240 else 175, 0.995
else:
    raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')

if not test:
    cutList.update({
        'Baseline':      Baseline,
        'Baseline_miss': Baseline_miss,
        'Baseline_sep':  Baseline + f' && ((visibleEnergy > {E_vis}) || (visibleEnergy < {E_vis} && cosTheta_miss < {theta_miss}))',
        'Baseline_vis':  Baseline + f' && visibleEnergy > {E_vis}',
        'Baseline_inv':  Baseline + f' && visibleEnergy < {E_vis}',
    })
cutList = {sel:cuts for sel, cuts in cutList.items() if (sel in sels or 'all' in sels)}



# List of selections to split into high/low BDT score regions
hl_default = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'Baseline_vis', 'Baseline_inv', 'test']
hl_sels    = list(cutList.keys()) if arg.hl_include=='all' else hl_default + arg.hl_include.split('-')

# Split each selection into high and low BDT score regions
cutList = make_high_low(cutList, bdt_cut, hl_sels)



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Custom histogram made at the pre-selection
customHists = custom_hists_ll if cat in ['ee', 'mumu'] else custom_hists_qq

# Output histogram definitions (name, title, binning)
histoList = histos_ll if cat in ['ee', 'mumu'] else histos_qq
histoList['BDTscore'] = {'name':'BDTscore', 'title':'BDT score', 'bin':1000,'xmin':0,'xmax':1}
if cat == 'qq':
    histoList['zqq_m_recoil_m_mva_high']      = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(50, 100, 150), (100, 40, 140), (1, bdt_cut, 1)]}
    histoList['zqq_m_recoil_m_mva_low']       = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(50, 100, 150), (100, 40, 140), (1, 0, bdt_cut)]}
    histoList['zqq_m_recoil_m_tot_mva_high']  = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(150, 50, 200), (100, 40, 140), (1, bdt_cut, 1)]}
    histoList['zqq_m_recoil_m_tot_mva_low']   = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(150, 50, 200), (100, 40, 140), (1, 0, bdt_cut)]}
    histoList['zqq_m_recoil_m_full_mva_high'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(350, 0, 350),  (180, 20, 200), (1, bdt_cut, 1)]}
    histoList['zqq_m_recoil_m_full_mva_low']  = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'], 'bins':[(350, 0, 350),  (180, 20, 200), (1, 0, bdt_cut)]}
