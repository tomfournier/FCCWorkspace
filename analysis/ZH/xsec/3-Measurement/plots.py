import time, argparse
t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat',  help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm',  help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', 
                    choices=[10.8, 3.1], type=float, default=10.8)

parser.add_argument('--yields',  help='Do not make yields plots',            action='store_true')
parser.add_argument('--decay',   help='Do not make Higgs decays only plots', action='store_true')
parser.add_argument('--make',    help='Do not make distribution plots',      action='store_true')
parser.add_argument('--sign',    help='Make significance scan plots',        action='store_true')
parser.add_argument('--cutflow', help='Make cutflow plots',                  action='store_true')

parser.add_argument('--tot',    help='Include all the Z decays in the plots', action='store_true')
arg = parser.parse_args()

import importlib
userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, z_decays, h_decays

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from tools.plotting import makePlot, PlotDecays, significance, AAAyields
from tools.cutflow import get_cutflow

ecm, lumi = arg.ecm, arg.lumi
categories = [arg.cat] if arg.cat!='' else ['ee', 'mumu']

def run(categories, selections, variables, procs_cfg, colors, legend, sign, tot):
    for cat in categories:
        print(f'\n----->[Info] Making plots for {cat} channel\n')
        procs = [f"Z{cat}H" if not tot else "ZH", "WW", "ZZ", "Zgamma", "Rare"] # first must be signal
        cat_decays = [cat] if not tot else z_decays

        for sel in selections:
            print(f'\n----->[Info] Making plots for {sel} selection\n')

            inputDir = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')
            outDir   = get_loc(loc.PLOTS_MEASUREMENT, cat, ecm, sel)

            if arg.cutflow and sel in cuts:
                inDir = get_loc(loc.EVENTS, cat, ecm, '')
                get_cutflow(inDir, outDir, cat, sel, procs, procs_cfg, colors, legend, cuts[sel], cuts_label[sel], 
                            z_decays, h_decays, sig_scale=1, defineList=defineList, tot=arg.tot)

            if not arg.yields: AAAyields('zll_recoil_m', inputDir, outDir, plots, legend, colors, cat, sel)
            if not arg.make and not arg.decay:
                for variable in variables:
                    print(f'\n----->[Info] Making plots for {variable}')

                    if sign:
                        significance(variable, variables, inputDir, outDir, procs_cfg, procs, sel=sel, reverse=True)
                        significance(variable, variables, inputDir, outDir, procs_cfg, procs, sel=sel, reverse=False)

                    if not arg.decay: PlotDecays(variable, variables, inputDir, outDir, cat_decays, h_decays, sel=sel)
                    if not arg.make:  makePlot(variable,   variables, inputDir, outDir, procs, procs_cfg, colors, legend, sel=sel)

selections = [
    'Baseline', 
    # 'Baseline_vis', 'Baseline_inv', 
    'Baseline_high', 'Baseline_low'
]

procs_cfg = {
"ZH"        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_decays for y in h_decays],
"ZmumuH"    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
"ZeeH"      : [f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_decays],
"WW"        : [f'p8_ee_WW_ecm{ecm}',             f'p8_ee_WW_mumu_ecm{ecm}', f'p8_ee_WW_ee_ecm{ecm}'],
"ZZ"        : [f'p8_ee_ZZ_ecm{ecm}'],
"Zgamma"    : [f'wzp6_ee_tautau_ecm{ecm}',       f'wzp6_ee_mumu_ecm{ecm}',
               f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],
"Rare"      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
               f'wzp6_gaga_mumu_60_ecm{ecm}',    f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
               f'wzp6_gammae_eZ_Zee_ecm{ecm}',   f'wzp6_gaga_ee_60_ecm{ecm}', 
               f'wzp6_gaga_tautau_60_ecm{ecm}',  f'wzp6_ee_nuenueZ_ecm{ecm}'],
}

plots = {}
plots['signal'] = {
    'ZH':[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays]
}
plots['backgrounds'] = {
    'WW':[f'p8_ee_WW_ecm{ecm}', f'p8_ee_WW_ee_ecm{ecm}', f'p8_ee_WW_mumu_ecm{ecm}'],
    'ZZ':    [f'p8_ee_ZZ_ecm{ecm}'], 
    'Zgamma':[f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
              f'wzp6_ee_tautau_ecm{ecm}'],
    'Rare':  [f"wzp6_egamma_eZ_Zee_ecm{ecm}",    f"wzp6_gammae_eZ_Zee_ecm{ecm}",
              f"wzp6_egamma_eZ_Zmumu_ecm{ecm}",  f"wzp6_gammae_eZ_Zmumu_ecm{ecm}", 
              f"wzp6_gaga_ee_60_ecm{ecm}",       f"wzp6_gaga_mumu_60_ecm{ecm}",
              f"wzp6_gaga_tautau_ecm{ecm}",      f"wzp6_ee_nuenueZ_ecm{ecm}"]
}

# colors from https://github.com/mpetroff/accessible-color-cycles
colors = {}
colors['ZH']       = ROOT.TColor.GetColor("#e42536")
colors['ZmumuH']   = ROOT.TColor.GetColor("#e42536")
colors['ZeeH']     = ROOT.TColor.GetColor("#e42536")
colors['ZZ']       = ROOT.TColor.GetColor("#5790fc")
colors['WW']       = ROOT.TColor.GetColor("#f89c20")
colors['Zgamma']   = ROOT.TColor.GetColor("#964a8b")
colors['Rare']     = ROOT.TColor.GetColor("#9c9ca1")

legend = {}
legend['ZH']       = 'ZH'
legend['ZmumuH']   = 'Z(#mu^{+}#mu^{-})H'
legend['ZeeH']     = 'Z(e^{+}e^{-})H'
legend['ZZ']       = 'ZZ'
legend['WW']       = 'W^{+}W^{-}'
legend['Zgamma']   = 'Z/#gamma^{*} #rightarrow f#bar{f}+#gamma(#gamma)'
legend['Rare']     = 'Rare'

cuts = {}
cuts['Baseline'] =  {
    'cut0': 'return true;',
    'cut1': 'leps_no >= 1 && leps_sel_iso.size() > 0',
    'cut2': 'leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()',
    'cut3': 'zll_m > 86 && zll_m < 96',
    'cut4': 'zll_p > 20 && zll_p < 70'
}
cuts_label = {}
cuts_label['Baseline'] = {
    'cut0': 'No cut',
    'cut1': '#geq 1#ell^{#pm} + ISO',
    'cut2': '#geq 2 #ell^{#pm} + OS',
    'cut3': '86 < m_{#ell^{+}#ell^{-}} < 96 GeV',
    'cut4': '20 < p_{#ell^{+}#ell^{-}} < 70 GeV'
}

defineList = {}



variables = {

    # Leptons
    "leading_p":        {"lim": [40, 100, 0, -1],
                         "limH": [40, 100, 1e-5, 0.06],
                         "xlabel": "p_{l,leading} [GeV]",
                         "logY": False},

    "leading_theta":    {"lim": [0, 3.2, 1, -1],
                         "limH": [0, 3.2, 1e-5, 1],
                         "xlabel": "#theta_{l,leading}", 
                         "logY": True},

    "subleading_p":     {"lim": [20, 60, 0, -1],
                         "limH": [20, 60, 1e-5, 0.06],
                         "xlabel": "p_{l,subleading} [GeV]",
                         "logY": False},

    "subleading_theta": {"lim":[0, 3.2, 1, -1],
                         "limH": [0, 3.2, 1e-5, 1],
                         "xlabel":"#theta_{l,subleading}",
                         "logY": True},

    "leading_phi":      {"lim": [-3.2, 3.2, 0, -1],
                         "limH": [-3.2, 3.2, 1e-5, 0.06],
                         "xlabel": "#phi_{l,leading}",
                         "logY": False},

    "subleading_phi":   {"lim": [-3.2, 3.2, 0, -1],
                         "limH": [-3.2, 3.2, 1e-5, 0.06],
                         "xlabel": "#phi_{l,subleading}",
                         "logY": False},

    # Zed
    "zll_m":     {"lim": [86, 96, 1, -1],
                  "limH": [86, 96, 1e-3, 1],
                  "xlabel": "m_{l^{+}l^{-}} [GeV]",
                  "logY": True},

    "zll_p":     {"lim": [20, 70, 1, -1],
                  "limH": [20, 70, 1e-4, 10],
                  "xlabel": "p_{l^{+}l^{-}} [GeV]",
                  "logY": True},

    "zll_theta": {"lim": [0, 3.2, 1, -1],
                  "limH": [0, 3.2, 1e-5, 1],
                  "xlabel": "#theta_{l^{+}l^{-}}",
                  "logY": True},
    
    "zll_phi":   {"lim": [-3.2, 3.2, 0, -1],
                  "limH": [-3.2, 3.2, 1e-5, 0.06],
                  "xlabel": "#phi_{l^{+}l^{-}} [GeV]",
                  "logY": False},

    # more control variables
    "acolinearity": {"lim": [0, 3, 1, -1],
                     "limH": [0, 3, 1e-5, 1],
                     "xlabel": "#Delta#theta_{l^{+}l^{-}}",
                     "logY": True},

    "acoplanarity": {"lim": [0, 3.2, 1, -1],
                     "limH": [0, 3.2, 1e-5, 1],
                     "xlabel": "#pi-#Delta#phi_{l^{+}l^{-}}",
                     "logY": True},

    "zll_deltaR":   {"lim": [0, 6.5, 1, -1],
                     "limH": [0, 6.5, 1e-5, 10],
                     "xlabel": "#Delta R",
                     "logY": True},
    
    # Recoil
    "zll_recoil_m": {"lim": [100, 150, 1, -1],
                     "limH": [100, 150, 1e-5, 0.5],
                     "xlabel": "m_{recoil} [GeV]",
                     "logY": False},

    # missing Information
    "visibleEnergy": {"lim": [0, 160, 1, -1],
                      "limH": [0, 160, 1e-5, 10],
                      "xlabel": "E_{vis} [GeV]",
                      "logY": True},
    
    "cosTheta_miss": {"lim": [0.9, 1, 1, -1],
                      "limH": [0.9, 1, 1e-3, 10],
                      "xlabel": "cos#theta_{miss}",
                      "logY": True},
    
    "missingMass":   {"lim": [0, 150, 1, -1],
                      "limH": [0, 150, 1e-5, 10],
                      "xlabel": "m_{miss} [GeV]",
                      "logY": True},

    # Higgsstrahlungness
    "H":        {"lim": [0, 110, 1, -1],
                 "limH": [0, 110, 1e-4, 10],
                 "xlabel": "Higgsstrahlungness",
                 "logY": True},
    
    # BDT score
    "BDTscore": {"lim": [0, 1, 1, -1],
                 "limH": [0, 1, 1e-5, 1],
                 "xlabel": "BDT score",
                 "logY": True}
}



if __name__=='__main__':
    run(categories, selections, variables, procs_cfg, colors, legend, arg.sign, arg.tot)

    print('\n\n------------------------------------\n')
    print(f'Time taken to run the code: {time.time()-t1:.1f} s')
    print('\n------------------------------------\n\n')
