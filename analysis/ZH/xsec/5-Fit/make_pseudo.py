import os
import ROOT
import argparse
import importlib, time

t1 = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, z_decays, h_decays
from tools.bias import getHists, make_pseudodata, make_datacard

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

parser.add_argument("--combine", help='Combine the channel to do the fit', action='store_true')

parser.add_argument("--target",            help="Target pseudodata", 
                    type=str, default="bb")
parser.add_argument("--pert",              help="Target pseudodata size", 
                    type=float, default=1.0)
parser.add_argument("--run",               help="Run combine", action='store_true')
parser.add_argument("--freezeBackgrounds", help="Freeze backgrounds", action='store_true')
parser.add_argument("--floatBackgrounds",  help="Float backgrounds", action='store_true')
parser.add_argument("--plot_dc",           help="Plot datacard", action='store_true')

parser.add_argument("--polL", help="Scale to left polarization",  action='store_true')
parser.add_argument("--polR", help="Scale to right polarization", action='store_true')
parser.add_argument("--ILC",  help="Scale to ILC luminosity",     action='store_true')
arg = parser.parse_args()

cat, ecm, sel = arg.cat, arg.ecm, arg.sel

if arg.ILC: ## change fit to ASIMOV -t -1 !!!
    proc_scales  = {"ZH": 1.048, "WW": 0.971, "ZZ": 0.939, "Zgamma": 0.919,}
elif arg.polL:
    proc_scales  = {"ZH": 1.554, "WW": 2.166, "ZZ": 1.330, "Zgamma": 1.263,}
elif arg.polR:
    procs_scales = {"ZH": 1.047, "WW": 0.219, "ZZ": 1.011, "Zgamma": 1.018,}
else:
    proc_scales  = {}

inputDir = get_loc(loc.HIST_PROCESSED, cat, ecm, sel)
outDir   = get_loc(loc.BIAS_DATACARD,  cat, ecm, sel)

rebin, hists      = 1, []
# hName, categories = [f'{cat}_recoil_m_mva_vis', f'{cat}_recoil_m_mva_inv'], ['vis', 'inv']
hName, categories = [f'zll_recoil_m'], [f'z_{cat}']

# first must be signal
procs     = [f"ZH", "WW", "ZZ", "Zgamma", "Rare"]
procs_cfg = {
"ZH"        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_decays for y in h_decays],
"ZmumuH"    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
"ZeeH"      : [f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_decays],
"WW"        : [f'p8_ee_WW_ecm{ecm}'],
"ZZ"        : [f'p8_ee_ZZ_ecm{ecm}'],
"Zgamma"    : [f'wzp6_ee_tautau_ecm{ecm}',       f'wzp6_ee_mumu_ecm{ecm}',
               f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],
"Rare"      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
               f'wzp6_gaga_mumu_60_ecm{ecm}',    f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
               f'wzp6_gammae_eZ_Zee_ecm{ecm}',   f'wzp6_gaga_ee_60_ecm{ecm}', 
               f'wzp6_gaga_tautau_60_ecm{ecm}',  f'wzp6_ee_nuenueZ_ecm{ecm}'],
}


if not arg.combine:
    for i, categorie in enumerate(categories):
        for proc in procs:
            h = getHists(inputDir, hName[i], proc, procs_cfg)
            h.SetName(f"{categorie}_{proc}")
            hists.append(h)

        args = [inputDir, procs, procs_cfg, hName[i], arg.target, z_decays, h_decays]
        hist_pseudo = make_pseudodata(*args, ecm=ecm, variation=arg.pert)
        hist_pseudo.SetName(f"{categorie}_data_{arg.target}")
        hists.append(hist_pseudo)

    if not os.path.isdir(outDir):
        os.system(f'mkdir -p {outDir}')

    print('----->[Info] Saving pseudo histograms')
    fOut = ROOT.TFile(f"{outDir}/datacard_{arg.target}.root", "RECREATE")
    for hist in hists:
        hist.Write()
    fOut.Close()
    print(f'----->[Info] Histograms saved in {outDir}/datacard_{arg.target}.root')

    print('----->[Info] Making datacard')
    make_datacard(outDir, procs, arg.target, 1.01, categories,
                freezeBackgrounds=arg.freezeBackgrounds, floatBackgrounds=arg.floatBackgrounds, plot_dc=arg.plot_dc)

arg_cat, comb, arg_sel = f'--cat {arg.cat}' if arg.cat!='' else '', '--combine' if arg.combine else '', f'--sel {arg.sel}'

if arg.run:
    cmd = f"python3 5-Fit/fit.py {arg_cat} --bias --target {arg.target} --pert {arg.pert} {comb} {arg_sel}"
    os.system(cmd)
else:
    print('\n\n------------------------------------\n')
    print(f'Time taken to run the code: {time.time()-t1:.1f} s')
    print('\n------------------------------------\n\n')
