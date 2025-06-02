import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")
from userConfig import loc, ecm, final_state, samples_sig, combine

if combine:
    print('Do not use combine.py with the variable combine set to True in userConfig.py')
    exit(0)

intLumi = 1 # userConfig.intLumi * 1e6
mc_stats = False
rebin = 1

inputDir  = loc.HIST_PROCESSED
outputDir = loc.NOMINAL_DATACARD

# Signal samples for measurement
z_decays = ['bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu']
h_decays = ['bb', 'cc', 'ss', 'gg', 'mumu', 'tautau', 'WW', 'ZZ', 'Za', 'aa'] #, 'inv']

samples_sig = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]
samples_sig.append(f'wzp6_ee_ZH_Hinv_ecm{ecm}')


sig_procs = {'sig': samples_sig}

ee_ll = f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if final_state=='ee' else f'wzp6_ee_mumu_ecm{ecm}'
bkg_procs = {'ZZ':[f'p8_ee_ZZ_ecm{ecm}'],
             'WW': [f'p8_ee_WW_ecm{ecm}'],
             'Zgamma':[ee_ll, f'wzp6_ee_tautau_ecm{ecm}'],
             'Rare':[f'wzp6_egamma_eZ_Z{final_state}_ecm{ecm}', f'wzp6_gammae_eZ_Z{final_state}_ecm{ecm}',
                     f'wzp6_gaga_{final_state}_60_ecm{ecm}', f'wzp6_gaga_tautau_60_ecm{ecm}',
                     f'wzp6_ee_nuenueZ_ecm{ecm}']
}

categories = [f'z_{final_state}']


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
