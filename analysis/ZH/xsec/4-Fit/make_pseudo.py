import os
import ROOT
import argparse
import importlib, time

t1 = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, z_decays, h_decays
from tools.bias import getHists, unroll, make_pseudodata, make_datacard

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
parser.add_argument("--combine", help='Combine the channel to do the fit', action='store_true')

parser.add_argument("--target", type=str, help="Target pseudodata", default="bb")
parser.add_argument("--run", help="Run combine", action='store_true')
parser.add_argument("--pert", type=float, help="Target pseudodata size", default=1.0)
parser.add_argument("--freezeBackgrounds", help="Freeze backgrounds", action='store_true')
parser.add_argument("--floatBackgrounds", help="Float backgrounds", action='store_true')
parser.add_argument("--plot_dc", help="Plot datacard", action='store_true')

parser.add_argument("--polL", help="Scale to left polarization", action='store_true')
parser.add_argument("--polR", help="Scale to right polarization", action='store_true')
parser.add_argument("--ILC", help="Scale to ILC luminosity", action='store_true')
arg = parser.parse_args()

final_state, ecm = arg.cat, arg.ecm
sel = select(arg.recoil120, arg.miss, arg.bdt)

if arg.ILC: ## change fit to ASIMOV -t -1 !!!
    proc_scales  = {"ZH": 1.048, "WW": 0.971, "ZZ": 0.939, "Zgamma": 0.919,}
elif arg.polL:
    proc_scales  = {"ZH": 1.554, "WW": 2.166, "ZZ": 1.330, "Zgamma": 1.263,}
elif arg.polR:
    procs_scales = {"ZH": 1.047, "WW": 0.219, "ZZ": 1.011, "Zgamma": 1.018,}
else:
    proc_scales = {}

inputDir = get_loc(loc.HIST_PREPROCESSED, final_state, ecm, sel)
outDir   = get_loc(loc.BIAS_DATACARD, final_state, ecm, sel)

rebin, hists = 1, []
hName = f'{final_state}_recoil_m_mva'

# first must be signal
procs = [f"ZH", "WW", "ZZ", "Zgamma", "Rare"]
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
procs_cfg['ZH'].remove(f'wzp6_ee_nunuH_Hinv_ecm{ecm}')

if not arg.combine:
    for proc in procs:
        h = getHists(inputDir, hName, proc, procs_cfg)
        h = unroll(h, rebin=rebin)
        h.SetName(f"{final_state}_{proc}")
        hists.append(h)

    hist_pseudo = make_pseudodata(inputDir, procs, procs_cfg, hName, arg.target, 
                                  z_decays, h_decays, ecm=ecm, variation=arg.pert)
    hist_pseudo = unroll(hist_pseudo, rebin=rebin)
    hist_pseudo.SetName(f"{final_state}_data_{arg.target}")
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
    make_datacard(outDir, procs, final_state, arg.target, 1.01, 
                freezeBackgrounds=arg.freezeBackgrounds, floatBackgrounds=arg.floatBackgrounds, plot_dc=arg.plot_dc)

cat, comb = f'--cat {arg.cat}' if arg.cat!='' else '', '--combine' if arg.combine else ''

if arg.run:
    cmd = f"python3 4-Fit/fit.py {cat} --bias --target {arg.target} --pert {arg.pert} {comb}"
    os.system(cmd)
else:
    print('\n\n------------------------------------\n')
    print(f'Time taken to run the code: {time.time()-t1:.1f} s')
    print('\n------------------------------------\n\n')
