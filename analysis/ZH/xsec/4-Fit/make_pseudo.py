import os
import ROOT
import argparse
import importlib, time

t1 = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, final_state, procs_cfg, z_decays, h_decays, ecm
from tools.bias import getHists, unroll, make_pseudodata, make_datacard

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, help="Target pseudodata", default="bb")
parser.add_argument("--run", help="Run combine", action='store_true')
parser.add_argument("--pert", type=float, help="Target pseudodata size", default=1.0)
parser.add_argument("--freezeBackgrounds", help="Freeze backgrounds", action='store_true')
parser.add_argument("--floatBackgrounds", help="Float backgrounds", action='store_true')
parser.add_argument("--plot_dc", help="Plot datacard", action='store_true')

parser.add_argument("--polL", help="Scale to left polarization", action='store_true')
parser.add_argument("--polR", help="Scale to right polarization", action='store_true')
parser.add_argument("--ILC", help="Scale to ILC luminosity", action='store_true')
args = parser.parse_args()

if args.ILC: ## change fit to ASIMOV -t -1 !!!
    proc_scales = {"ZH": 1.048, "WW": 0.971, "ZZ": 0.939, "Zgamma": 0.919,}
elif args.polL:
    proc_scales = {"ZH": 1.554, "WW": 2.166, "ZZ": 1.330, "Zgamma": 1.263,}
elif args.polR:
    procs_scales = {"ZH": 1.047, "WW": 0.219, "ZZ": 1.011, "Zgamma": 1.018,}
else:
    proc_scales = {}

inputDir = loc.HIST_PREPROCESSED
outDir   = loc.BIAS_DATACARD

hName = f'{final_state}_recoil_m_mva'

rebin, hists = 1, []

# first must be signal
procs = [f"ZH", "WW", "ZZ", "Zgamma", "Rare"]

for proc in procs:
    h = getHists(inputDir, hName, proc, procs_cfg)
    h = unroll(h, rebin=rebin)
    h.SetName(f"{final_state}_{proc}")
    hists.append(h)

hist_pseudo = make_pseudodata(inputDir, procs, procs_cfg, hName, args.target, z_decays, h_decays, ecm=ecm, variation=args.pert)
hist_pseudo = unroll(hist_pseudo, rebin=rebin)
hist_pseudo.SetName(f"{final_state}_data_{args.target}")
hists.append(hist_pseudo)

if not os.path.isdir(outDir):
    os.system(f'mkdir -p {outDir}')

print('----->[Info] Saving pseudo histograms')
fOut = ROOT.TFile(f"{outDir}/datacard_{args.target}.root", "RECREATE")
for hist in hists:
    hist.Write()
fOut.Close()
print(f'----->[Info] Histograms saved in {outDir}/datacard_{args.target}.root')

print('----->[Info] Making datacard')
make_datacard(outDir, procs, final_state, args.target, 1.01, 
              freezeBackgrounds=args.freezeBackgrounds, floatBackgrounds=args.floatBackgrounds, plot_dc=args.plot_dc)

if args.run:
    cmd = f"python3 4-Fit/fit.py --bias --target {args.target} --pert {args.pert}"
    os.system(cmd)
else:
    print('\n\n------------------------------------\n')
    print(f'Time taken to run the code: {time.time()-t1:.1f} s')
    print('\n------------------------------------\n\n')
