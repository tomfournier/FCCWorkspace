import os, time, argparse, importlib

from ROOT import TFile

t = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat',  help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='ee-mumu')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)

parser.add_argument('--polL', help='Scale to left polarization',  action='store_true')
parser.add_argument('--polR', help='Scale to right polarization', action='store_true')
parser.add_argument('--ILC',  help='Scale to ILC luminosity',     action='store_true')
arg = parser.parse_args()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, add_package_path, get_loc
add_package_path(loc.PACKAGE)

from package.config import timer, mk_processes, z_decays, H_decays
from package.tools.utils import mkdir
from package.tools.process import get_hist, concat

cats, ecm = arg.cat.split('-'), arg.ecm
hNames = ['zll_recoil_m']
sels = [
    'Baseline',
    # 'Baseline_miss',
    # 'Baseline_sep',
    'Baseline_tight'
]

if arg.ILC: ## change fit to ASIMOV -t -1 !!!
    procs_scales  = {'ZH': 1.048, 'WW': 0.971, 'ZZ': 0.939, 'Zgamma': 0.919}
elif arg.polL:
    procs_scales  = {'ZH': 1.554, 'WW': 2.166, 'ZZ': 1.330, 'Zgamma': 1.263}
elif arg.polR:
    procs_scales = {'ZH': 1.047, 'WW': 0.219, 'ZZ': 1.011, 'Zgamma': 1.018}
else:
    procs_scales  = {}
if procs_scales!={}:
    print(f'----->[Info] Will scale histograms to ILC cross section')


processes = mk_processes(['ZH', 'WW', 'ZZ', 'Zgamma', 'Rare'], 
                         H_decays=H_decays+['ZZ_noInv'], 
                         ecm=ecm)

samples_bkg = [
    # ee -> WW
    f'p8_ee_WW_ecm{ecm}', 
    f'p8_ee_WW_ee_ecm{ecm}', 
    f'p8_ee_WW_mumu_ecm{ecm}', 

    # ee -> ZZ
    f'p8_ee_ZZ_ecm{ecm}',

    # ee -> Z/ga -> ll
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', 
    f'wzp6_ee_mumu_ecm{ecm}', 
    f'wzp6_ee_tautau_ecm{ecm}',

    # e ga -> e Z(ll)
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', 
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}',   
    f'wzp6_gammae_eZ_Zee_ecm{ecm}',

    # ga ga -> ll
    f'wzp6_gaga_ee_60_ecm{ecm}', 
    f'wzp6_gaga_mumu_60_ecm{ecm}', 
    f'wzp6_gaga_tautau_60_ecm{ecm}',

    # ee -> nu nu Z 
    f'wzp6_ee_nuenueZ_ecm{ecm}'
]
samples_sig = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in H_decays + ['ZZ_noInv']]
samples_sig.extend([f'wzp6_ee_eeH_ecm{ecm}', f'wzp6_ee_mumuH_ecm{ecm}', f'wzp6_ee_ZH_Hinv_ecm{ecm}'])

samples = samples_sig + samples_bkg



def run(cats: str, 
        sels: str, 
        hNames: list[str], 
        samples: list[str]
        ) -> None:
    
    for cat in cats:
        print(f'----->[Info] Processing histograms for {cat}')
        inDir = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')

        for sel in sels:
            print(f'\n----->[Info] Processing histograms for {sel}\n')
            outDir = get_loc(loc.HIST_PROCESSED, cat, ecm, sel)
            mkdir(outDir)

            suffix_high = f'_{sel}_high_histo'
            suffix_low  = f'_{sel}_low_histo'
            suffix_base = f'_{sel}_histo'

            existing_samples = []
            for sample in samples:
                if os.path.exists(f'{inDir}/{sample}{suffix_base}.root'):
                    existing_samples.append(sample)

            if not existing_samples:
                print(f'----->[Warning] No sampples found for {sel} in {cat}')
                continue

            for sample in existing_samples:
                print(f'----->[Info] Processing {sample}')
                hists = []
                for hName in hNames:
                    h_high = get_hist(hName, sample, processes, inDir, 
                                      suffix=suffix_high, 
                                      proc_scales=procs_scales)
                    h_low  = get_hist(hName, sample, processes, inDir,
                                      suffix=suffix_low,
                                      proc_scales=procs_scales)
                    h_high.SetName(h_high.GetName()+'_high')
                    h_low.SetName(h_low.GetName()+'_low')
                    
                    if h_high is None or h_low is None:
                        continue

                    has_valid_hist = True
                    h = concat([h_low, h_high], hName)

                    hists.extend([h, h_high, h_low])

                if not has_valid_hist:
                    print(f'----->[WARNING] No valid histograms found for {sample}, skipping file creation')
                    continue

                fOut = os.path.join(outDir, sample+'.root')
                f = TFile(fOut, 'RECREATE')
                for hist in hists:
                    hist.Write()
                f.Close()
                print(f'----->[Info] Saved histograms in {fOut}\n')

if __name__=='__main__':
    run(cats, sels, hNames, samples)
    timer(t)
