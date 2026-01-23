##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, json

from pathlib import Path

# Load userConfig
from package.userConfig import (
    loc, get_loc, event
)
from package.config import z_decays, H_decays

# Load config from temporary JSON if running automated, else prompt
if os.environ.get('RUN'):
    cfg_file = Path(loc.RUN) / '4-run.json'
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text())
        cat, ecm, sel = cfg['cat'], cfg['ecm'], cfg['sel']
        print(f'Loaded config: {cat = }, {ecm = }, {sel = }')
    else:
        raise FileNotFoundError(f"Couldn't find {cfg_file} file")
else:
    cat = input('Select channel [ee, mumu]: ')
    sel = input('Select a selection: ')
    from package.userConfig import ecm

if cat not in ['ee', 'mumu']:
    raise ValueError(f'Invalid channel: {cat}. Must be "ee" or "mumu"')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Datacard configuration parameters
mc_stats = False  # Include MC statistical uncertainties
rebin    = 1      # Histogram rebinning factor
intLumi  = 1      # Integrated luminosity scale factor

# Define input and output directories
inputDir  = get_loc(loc.HIST_PROCESSED,   cat, ecm, sel)
outputDir = get_loc(loc.NOMINAL_DATACARD, cat, ecm, sel)
inDir     = get_loc(loc.EVENTS,           cat, ecm, '')

# Define signal processes: ee -> Z(ll)H with various Higgs decay modes
if sel=='Jan_sample':
    samples_sig = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in H_decays]
else:
    samples_sig = event([f'wzp6_ee_{x}H_H{y}_ecm{ecm}'.replace('HZZ', 'HZZ_noInv') for x in z_decays for y in H_decays], inDir)
sig_procs = {'sig': samples_sig}

# Define background processes: ZZ, WW, Z/gamma, and rare processes
bkg_procs = {
    'ZZ':    event([f'p8_ee_ZZ_ecm{ecm}'], inDir),
    'WW':    event([f'p8_ee_WW_ecm{ecm}', 
                    f'p8_ee_WW_ee_ecm{ecm}', 
                    f'p8_ee_WW_mumu_ecm{ecm}'], inDir),
    'Zgamma':event([f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', 
                    f'wzp6_ee_mumu_ecm{ecm}', 
                    f'wzp6_ee_tautau_ecm{ecm}'], inDir),
    'Rare':  event([f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}', 
                    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
                    f'wzp6_gaga_{cat}_60_ecm{ecm}', 
                    f'wzp6_gaga_tautau_60_ecm{ecm}',
                    f'wzp6_ee_nuenueZ_ecm{ecm}'], inDir)
}

# Define histogram names
hName = [
    'zll_recoil_m' if sel!='Jan_sample' else f'{cat}_zll_recoil_m_mva'
]
# Configure categories based on fit type: single category or visible/invisible split
categories = [f'z_{cat}']
hist_names = [hName[0]]

# Define systematic uncertainties: normalization uncertainties for backgrounds
systs = {}
for i in sorted(['WW', 'ZZ', 'Zgamma', 'Rare']):
    systs[f'{i}_norm'] = {
        'type':  'lnN',       # Log-normal uncertainty
        'value': 1.01,        # 1% normalization uncertainty
        'procs': [i]          # Apply to this background process
    }
