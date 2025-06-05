import importlib, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
arg = parser.parse_args()

if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)

# Load userConfig
userConfig = importlib.import_module("userConfig")
from userConfig import loc, get_loc, select, z_decays, h_decays

final_state, ecm = arg.cat, arg.ecm
sel = select(arg.recoil120, arg.miss, arg.bdt)

intLumi = 1 # userConfig.intLumi * 1e6
mc_stats = False
rebin = 1

inputDir  = get_loc(loc.HIST_PROCESSED, final_state, ecm, sel)
outputDir = get_loc(loc.NOMINAL_DATACARD, final_state, ecm, sel)

samples_sig = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]
# samples_sig.append(f'wzp6_ee_ZH_Hinv_ecm{ecm}')

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
