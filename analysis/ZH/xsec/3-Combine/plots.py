import time, argparse
t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', choices=[10.8, 3.1], type=float, default=10.8)

parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
parser.add_argument('--leading', help='Add the p_leading and p_subleading cuts', action='store_true')
parser.add_argument('--vis', help='Add a cut on visible energy', action='store_true')
parser.add_argument('--visbdt', help='Add E_vis in the training variables for the BDT', action='store_true')
parser.add_argument('--sep', help='Separate events by using E_vis', action='store_true')

parser.add_argument('--sign', help='Make significance plots', action='store_true')
arg = parser.parse_args()

import importlib
userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, z_decays, h_decays

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from tools.plotting import makePlot, CutFlow, PlotDecays, CutFlowDecays, significance

ecm, lumi = arg.ecm, arg.lumi
m_dw, m_up = '120' if arg.recoil120 else '100', '140' if arg.recoil120 else '150'
cuts = ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]
cut_labels = ["All events", "#geq 1 #ell^{#pm} + ISO", "#geq 2 #ell^{#pm} + OS", "86 < m_{#ell^{+}#ell^{#minus}} < 96", 
              "20 < p_{#ell^{+}#ell^{#minus}} < 70", m_dw+" < m_{rec} < "+m_up]
if arg.leading:
        cuts.append('cut6')
        cut_labels.append("50<p_{leading}<80 and p_{sub}<53")
if arg.sep:
     cuts.append('cut7')
     cut_labels.append('E_{vis} separation')
if arg.vis:
        cuts.append('cut8')
        cut_labels.append("E_{vis} > 10")
if arg.miss:
        cu = 'cut9' if arg.vis else 'cut8'
        cuts.append(cu)
        cut_labels.append("|cos#theta_{miss}| < 0.98")

procs_cfg = {
"ZH"        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_decays for y in h_decays],
"ZmumuH"    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
"ZeeH"      : [f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_decays],
"WW"        : [f'p8_ee_WW_ecm{ecm}'],
"ZZ"        : [f'p8_ee_ZZ_ecm{ecm}'],
"Zgamma"    : [f'wzp6_ee_tautau_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
               f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],
"Rare"      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
               f'wzp6_gaga_mumu_60_ecm{ecm}',    f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
               f'wzp6_gammae_eZ_Zee_ecm{ecm}',   f'wzp6_gaga_ee_60_ecm{ecm}', 
               f'wzp6_gaga_tautau_60_ecm{ecm}',  f'wzp6_ee_nuenueZ_ecm{ecm}'],
}

argument = {
    'zll_m_nOne': {
        'lim':   [50, 120, 1e2, 1e8],
        'limH':  [15, 130, 1e-5, 1],
        'label': ['m_{ll} [GeV]', 'Events'],
        'logY':  True, 'rebin': 1,
        'sign':  True
    },

    'zll_p_nOne': {
        'lim':   [0, 120, 1e1, 1e8],
        'limH':  [0, 80, 1e-5, 1],
        'label': ['p_{ll} [GeV]', 'Events'],
        'logY':  True, 'rebin': 2,
        'sign':  True
    },

    'cosThetaMiss_nOne': {
        'lim':   [0.9, 1, 1e1, 1e7],
        'limH':  [0.9, 1, 1e-5, 1e1],
        'label': ['|cos#theta_{miss}|', 'Events'],
        'logY':  True, 'rebin': 8,
        'sign':  True
    },

    'zll_recoil_m_mva_low': {
        'lim':   [int(m_dw), int(m_up), 0, -1],
        'limH':  [int(m_dw), int(m_up), 1e-5, 1],
        'label': ['m_{recoil} [GeV]', 'Events'],
        'logY':  False, 'rebin': 4
    },

    'zll_recoil_m_mva_high': {
        'lim':   [122, 134, 0, -1],
        'limH':  [122, 134, 1e-5, 1e1],
        'label': ['m_{recoil} [GeV]', 'Events'],
        'logY':  False, 'rebin': 1
    },

    'mva_score': {
        'lim':   [0, 1, 1e-1, -1],
        'limH':  [0, 1, 1e-4, 1],
        'label': ['MVA score', 'Events'],
        'logY':  True, 'rebin': 10,
        'sign':  True
    },

    'zll_recoil_nOne': {
        'lim':   [100, 150, 1, 1e6],
        'limH':  [100, 150, 1e-5, 1],
        'label': ['m_{recoil} [GeV]', 'Events'],
        'logY':  True, 'rebin': 8,
        'sign':  True
    },

    'zll_recoil_cut3': {
        'lim':   [40, 160, 0, 40e3],

        'label': ['m_{recoil} [GeV]', 'Events'],
        'logY':  False, 'rebin': 8
    },

    'zll_recoil_cut4': {
        'lim':   [40, 160, 0, 1e4],
        'label': ['m_{recoil} [GeV]', 'Events'],
        'logY':  False, 'rebin': 8
    },

    'leps_iso_noSel': {
        'lim':   [0, 5, 1e-1, -1],
        'limH':  [0, 4, 1e-5, 1],
        'label': ['Cone Isolation', 'Events'],
        'logY':  True, 'rebin': 1,
        'sign':  True
    },

    'leps_all_p_noSel': {
        'lim':   [20, 100, 1e1, -1],
        'limH':  [20, 100, 1e-5, 1],
        'label': ['p_{lepton} no sel [GeV]', 'Events'],
        'logY':  True, 'rebin': 1,
        'sign':  True
    },

    'leps_all_theta_noSel': {
        'lim':   [0, 3.2, 1e1, -1],
        'limH':  [0, 3.2, 1e-5, 1],
        'label': ['#theta_{lepton} no sel [GeV]', 'Events'],
        'logY':  True, 'rebin': 1,
        'sign':  True
    },

    'visibleEnergy_nOne': {
        'lim':   [0, 160, 1e-1, -1],
        'limH':  [0, 160, 1e-5, 1], 
        'label': ['E_{vis} [GeV]', 'Events'],
        'logY':  True, 'rebin': 16,
        'sign':  True
    },

    'zll_m': {
        'lim':   [86, 96, 1e1, -1],
        'limH':  [86, 96, 1e-4, 1],
        'label': ['m_{ll} [GeV]', 'Events'],
        'logY':  True, 'rebin': 1,
        'hl':    True
    },

    'zll_p': {
        'lim':   [20, 70, 1e1, -1],
        'limH':  [20, 70, 1e-4, 1],
        'label': ['p_{ll} [GeV]', 'Events'],
        'logY':  True, 'rebin': 1,
        'hl':    True
    },

    'zll_recoil': {
        'lim':   [int(m_dw), int(m_up), 0, -1],
        'limH':  [int(m_dw), int(m_up), 1e-4, 1],
        'label': ['m_{recoil} [GeV]', 'Events'],
        'logY':  False, 'rebin': 8,
        'hl':    True
    },

    'acoplanarity': {
        'lim':   [0, 3.2, 1e-2, 1e7],
        'limH':  [0, 3.2, 1e-5, 1],
        'label': ['#Delta#phi_{ll}', 'Events'],
        'logY':  True, 'rebin': 1,
        'hl':    True, 'sign':  True
    },

    'acolinearity': {
        'lim':   [0, 3.2, 1e-2, 1e7],
        'limH':  [0, 3.2, 1e-5, 1],
        'label': ['#Delta#theta_{ll}', 'Events'],
        'logY':  True, 'rebin': 1,
        'hl':    True, 'sign':  True
    },

    'leading_p': {
        'lim':   [40, 100, 1, -1],
        'limH':  [40, 90, 1e-5, 1e1],
        'label': ['p_{leading} [GeV]', 'Events'],
        'logY':  True, 'rebin': 1,
        'hl':    True, 'sign':  True
    },

    'subleading_p': {
        'lim':   [10, 70, 1, -1],
        'limH':  [20, 60, 1e-5, 1e1],
        'label': ['p_{subleading} [GeV]', 'Events'],
        'logY':  True, 'rebin': 4,
        'hl':    True, 'sign':  True
    },

     'zll_theta': {
        'lim':   [0, 3.2, 1e1, -1],
        'limH':  [0, 3.2, 1e-5, 1],
        'label': ['#theta_{ll}', 'Events'],
        'logY':  True, 'rebin': 2,
        'sign':  True,  'hl': True
    },

    'leading_theta': {
        'lim':   [0, 3.2, 1e1, 1e7],
        'limH':  [0, 3.2, 1e-4, 1],
        'label': ['#theta_{leading}', 'Events'],
        'logY':  True, 'rebin': 4,
        'hl':    True, 'sign':  True
    },

    'subleading_theta': {
        'lim':   [0, 3.2, 1e1, 1e6],
        'limH':  [0, 3.2, 1e-4, 1],
        'label': ['#theta_{subleading}', 'Events'],
        'logY':  True, 'rebin': 4,
        'hl':    True, 'sign':  True
    }, 

    'leps_p': {
        'lim':   [20, 100, 1e-2, 1e9],
        'limH':  [20, 90, 1e-5, 1], 
        'label': ['p_{leptons} [GeV]', 'Events'],
        'logY':  True, 'rebin': 4,
        'hl':    True, 'sign':  True
    },

    'visibleEnergy_nOne': {
        'lim':   [0, 160, 1e-1, -1],
        'limH':  [0, 160, 1e-5, 1], 
        'label': ['E_{vis} [GeV]', 'Events'],
        'logY':  True, 'rebin': 16,
        'sign':  True
    },    

    'visibleEnergy': {
        'lim':   [10, 160, 1e-2, -1],
        'limH':  [10, 160, 1e-5, 1],
        'label': ['E_{vis} [GeV]', 'Events'],
        'logY':  True, 'rebin': 2,
        'hl':    True, 'sign':  True
    }          
}


for cat in ['ee', 'mumu']:
    print(f'\n----->[Info] Making plots for {cat} channel\n')

    sel      = select(arg.recoil120, arg.miss, arg.bdt, arg.leading, arg.vis, arg.visbdt, arg.sep)
    inputDir = get_loc(loc.HIST_PREPROCESSED, cat, ecm, sel)
    outDir   = get_loc(loc.PLOTS_MEASUREMENT, cat, ecm, sel)

    procs = [f"ZH", "WW", "ZZ", "Zgamma", "Rare"] # first must be signal

    lep = 'e' if cat=='ee' else '#mu'
    for i, cut in enumerate(cut_labels): cut_labels[i] = cut.replace('#ell', lep)

    args = [inputDir, outDir, procs, procs_cfg]
    args1 = [inputDir, outDir, [cat], h_decays]

    CutFlow(*args, hName=f"{cat}_cutFlow", cuts=cuts, labels=cut_labels, outName='cutflow', 
            sig_scale=10, yMin=1e4, yMax=1e10)
    CutFlowDecays(*args1[:2], cat, hName=f"{cat}_cutFlow", outName="cutFlow", 
            cuts=cuts, cut_labels=cut_labels, yMin=40, yMax=150, z_decays=[cat], h_decays=h_decays, miss=arg.miss)
    CutFlowDecays(*args1[:2], cat, hName=f"{cat}_cutFlow", outName="cutFlow_all", 
            cuts=cuts, cut_labels=cut_labels, yMin=40, yMax=150, z_decays=z_decays, h_decays=h_decays, miss=arg.miss)

    for hName, kwarg in argument.items():
        print(f'\n----->[Info] Making plots for {hName}')
        for ind in ['', '_high', '_low', '_vis', '_inv', '_mvis', '_minv']:
            if ind!='' and not 'hl' in kwarg: continue
            if arg.sign and 'sign' in kwarg and kwarg['sign']:
                significance(f'{cat}_{hName}{ind}', *args, *kwarg['lim'][:2], outName=hName, reverse=True)
                significance(f'{cat}_{hName}{ind}', *args, *kwarg['lim'][:2], outName=hName, reverse=False)

            i, cond = ind if ind!='' else '_nominal', (ind=='') or (ind=='_high') or (ind=='_low') or (ind=='_vis')
            reb = kwarg['rebin'] if cond else 8
            print(f'\n----->[Info] Making plots for {i[1:]}')
            if 'limH' in kwarg: 
                 PlotDecays(f'{cat}_{hName}{ind}', *args1, *kwarg['limH'], *kwarg['label'], 
                            outName=hName, rebin=reb) 
            if 'lim' in kwarg: 
                 makePlot(f'{cat}_{hName}{ind}', *args, *kwarg['lim'], *kwarg['label'], 
                          outName=hName, logY=kwarg['logY'], rebin=kwarg['rebin'])
            


print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')