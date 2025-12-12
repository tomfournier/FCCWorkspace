import importlib

# Load userConfig
userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, event, ecm, fitting

from package.config import z_decays, h_decays

cat = input('Select a channel [ee, mumu]: ')
sel = input('Select a selection: ')

mc_stats = False
rebin    = 1
intLumi  = 1

inputDir  = get_loc(loc.HIST_PROCESSED,   cat, ecm, sel)
outputDir = get_loc(loc.NOMINAL_DATACARD, cat, ecm, sel)
inDir     = get_loc(loc.EVENTS,           cat, ecm, '')+'/'

samples_sig = event([f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays], inDir)
sig_procs = {'sig': samples_sig}

bkg_procs = {
    'ZZ':    event([f'p8_ee_ZZ_ecm{ecm}'], inDir),
    'WW':    event([f'p8_ee_WW_ecm{ecm}',              f'p8_ee_WW_ee_ecm{ecm}', 
                    f'p8_ee_WW_mumu_ecm{ecm}'], inDir),
    'Zgamma':event([f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}', 
                    f'wzp6_ee_tautau_ecm{ecm}'], inDir),
    'Rare':  event([f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}', f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
                    f'wzp6_gaga_{cat}_60_ecm{ecm}',    f'wzp6_gaga_tautau_60_ecm{ecm}',
                    f'wzp6_ee_nuenueZ_ecm{ecm}'], inDir)
}

hName = [
    'zll_recoil_m'
]
if not fitting:
    categories = [f'z_{cat}']
    hist_names = [hName[0]]
else:
    categories = [f'z_{cat}_vis', f'z_{cat}_inv']
    hist_names = [hName[1], hName[2]]

systs = {}
for i in ['Rare', 'WW', 'ZZ', 'Zgamma']:
    systs[f'{i}_norm'] = {
        'type':  'lnN',
        'value': 1.01,
        'procs': [i]
    }
