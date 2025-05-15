import importlib

# Load userConfig
userConfig = importlib.import_module("userConfig")

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

intLumi = 1
mc_stats = True
rebin = 1

inputDir = userConfig.loc.ANALYSIS_FINAL
outputDir = userConfig.loc.COMBINE

if userConfig.combine:
    sig_procs = {'sig':[f'wzp6_ee_eeH_ecm{ecm}', f'wzp6_ee_mumuH_ecm{ecm}']}

    bkg_procs = {'bkg':[f'p8_ee_ZZ_ecm{ecm}', 
                        f'p8_ee_WW_ecm{ecm}',
                        f'wzp6_egamma_eZ_Zee_ecm{ecm}', f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
                        f'wzp6_gammae_eZ_Zee_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
                        f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
                        f'wzp6_gaga_ee_60_ecm{ecm}', f'wzp6_gaga_mumu_60_ecm{ecm}',
                        f'wzp6_ee_tautau_ecm{ecm}',
                        f'wzp6_gaga_tautau_60_ecm{ecm}',
                        f'wzp6_ee_nuenueZ_ecm{ecm}']
    }
else:
    ee_ll = f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if final_state=='ee' else f'wzp6_ee_mumu_ecm{ecm}'
    sig_procs = {'sig':[f'wzp6_ee_{final_state}H_ecm{ecm}']}

    bkg_procs = {'bkg':[f'p8_ee_ZZ_ecm{ecm}', 
                        f'p8_ee_WW_ecm{ecm}',
                        f'wzp6_egamma_eZ_Z{final_state}_ecm{ecm}',
                        f'wzp6_gammae_eZ_Z{final_state}_ecm{ecm}',
                        ee_ll,
                        f'wzp6_gaga_{final_state}_60_ecm{ecm}',
                        f'wzp6_ee_tautau_ecm{ecm}',
                        f'wzp6_gaga_tautau_60_ecm{ecm}',
                        f'wzp6_ee_nuenueZ_ecm{ecm}']
    }


if userConfig.combine:
    categories = ['zee','zmumu']
else:
    categories = [f'z{final_state}']

if userConfig.miss:
    selection = 'sel_Baseline'
else:
    selection = 'sel_Baseline_no_costhetamiss'

hist_names = ['recoil_m']

systs = {}

systs['bkg_norm'] = {
    'type': 'lnN',
    'value': 1.10,
    'procs': ['bkg',]
}

systs['lumi'] = {
    'type': 'lnN',
    'value': 1.01,
    'procs': ['sig', 'bkg'],
}
