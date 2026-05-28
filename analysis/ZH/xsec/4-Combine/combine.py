##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, json

from pathlib import Path

# Load user configuration and event utilities
from package.userConfig import (
    loc,     # Directory path manager
    event    # Event sample selection
)
from package.config import z_decays, H_decays  # Physics process definitions

# Load configuration from JSON file if automated, otherwise prompt user
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
    ecm = input('Select center-of-mass energy [240, 365]: ')
    sel = input('Select a selection: ')

if cat not in ['ee', 'mumu']:
    raise ValueError(f'Invalid channel: {cat}. Must be "ee" or "mumu"')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Datacard configuration for combine tool
mc_stats = False  # Include MC statistical uncertainties (default: False, assume Poisson)
rebin    = 1      # Histogram rebinning factor (1 = no rebinning)
intLumi  = 1      # Luminosity scaling factor for normalization

# Define input/output directories
inputDir  = loc.get('HIST_PROCESSED',   cat, ecm, sel)  # Processed histograms from step 4
outputDir = loc.get('NOMINAL_DATACARD', cat, ecm, sel)  # Output datacards for combine
inDir     = loc.get('EVENTS_TEST',      cat, ecm)           # Event sample directory

# Define signal processes: ZH production with all Higgs decay modes
samples_sig = event([f'wzp6_ee_{x}H_H{y}_ecm{ecm}'.replace('HZZ', 'HZZ_noInv') for x in z_decays for y in H_decays], inDir)
sig_procs = {'sig': samples_sig}  # Combine all signal samples under 'sig' label

# Define background processes with their respective samples
bkg_procs = {
    'ZZ':     event([f'p8_ee_ZZ_ecm{ecm}'], inDir),      # Diboson ZZ
    'WW':     event([f'p8_ee_WW_ecm{ecm}',                    # Diboson WW
                     f'p8_ee_WW_ee_ecm{ecm}',
                     f'p8_ee_WW_mumu_ecm{ecm}'], inDir),
    'Zgamma': event([f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',       # Drell-Yan processes
                     f'wzp6_ee_mumu_ecm{ecm}',
                     f'wzp6_ee_tautau_ecm{ecm}'], inDir),
    'Rare':   event([f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',       # Rare backgrounds
                     f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
                     f'wzp6_gaga_{cat}_60_ecm{ecm}',
                     f'wzp6_gaga_tautau_60_ecm{ecm}',
                     f'wzp6_ee_nuenueZ_ecm{ecm}'], inDir)
}

# Define observable (histogram names) for fitting
hName = ['zll_recoil_m']
# Define fit categories (one per decay mode)
categories = [f'z_{cat}']   # Category identifier (e.g., 'z_ee', 'z_mumu')
hist_names = [hName[0]]     # Histogram name for this category

# Define systematic uncertainties
# Log-normal normalization uncertainties (1% each) for all background processes
systs = {}
for i in sorted(['WW', 'ZZ', 'Zgamma', 'Rare']):
    systs[f'{i}_norm'] = {
        'type':  'lnN',       # Log-normal uncertainty type
        'value': 1.01,        # 1% normalization uncertainty (±1%)
        'procs': [i]          # Apply to this background process
    }
