import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")
from userConfig import loc, ecm, final_state, samples_sig, combine

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

intLumi = 1 # userConfig.intLumi * 1e6
mc_stats = True
rebin = 1

inputDir  = loc.HIST_PROCESSED
outputDir = loc.NOMINAL_DATACARD

sig_procs = {'sig': samples_sig}

ee_ll = f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if final_state=='ee' else f'wzp6_ee_mumu_ecm{ecm}'
bkg_procs = {'ZZ':[f'p8_ee_ZZ_ecm{ecm}'],
             'WW': [f'p8_ee_WW_ecm{ecm}'],
             'Zgamma':[ee_ll, f'wzp6_ee_tautau_ecm{ecm}'],
             'Rare':[f'wzp6_egamma_eZ_Z{final_state}_ecm{ecm}', f'wzp6_gammae_eZ_Z{final_state}_ecm{ecm}',
                     f'wzp6_gaga_{final_state}_60_ecm{ecm}', f'wzp6_gaga_tautau_60_ecm{ecm}',
                     f'wzp6_ee_nuenueZ_ecm{ecm}']
}

categories = [f'z_{final_state}'] if not combine else ['z_lep']


hist_names = [f'{final_state}_recoil_m_mva']

systs = {}

systs['bkg_norm'] = {
    'type': 'lnN',
    'value': 1.01,
    'procs': ['ZZ','WW','Zgamma','Rare']
}

# systs['lumi'] = {
#     'type': 'lnN',
#     'value': 1.01,
#     'procs': ['sig', 'ZZ','WW','Zgamma','Rare'],
# }
