import time, importlib, argparse
t = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat',  help='Final state (ee, mumu)', 
                    choices=['ee', 'mumu', 'ee-mumu'], type=str, default='ee-mumu')
parser.add_argument('--ecm',  help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity (10.8, 3.1) in ab-1', 
                    choices=[10.8, 3.1], type=float, default=10.8)

parser.add_argument('--tot', help='Include all the Z decays in the plots', 
                    action='store_true')
arg = parser.parse_args()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, add_package_path
add_package_path(loc.PACKAGE)

from package.config import timer, mk_processes
from package.config import z_decays, H_decays, colors, labels
from package.plots.cutflow import get_cutflow, branches_from_cuts



cats, ecm, lumi = arg.cat.split('-'), arg.ecm, arg.lumi
sels = [
    'Baseline', 'Baseline_miss', 'Baseline_sep', 'Baseline_tight'
]

baseline_cuts = {
    'cut0': '', 'cut1': '', 'cut2': '',
    'cut3': 'zll_m > 86 & zll_m < 96',
    'cut4': 'zll_p > 20 & zll_p < 70'
}
baseline_labels = {
    'cut0': 'No cut',
    'cut1': '#geq 1#ell^{#pm} + ISO',
    'cut2': '#geq 2 #ell^{#pm} + OS',
    'cut3': '86 < m_{#ell^{+}#ell^{-}} < 96 GeV',
    'cut4': '20 < p_{#ell^{+}#ell^{-}} < 70 GeV'
}

cuts = {sel: baseline_cuts.copy() for sel in sels}
cuts_label = {sel: baseline_labels.copy() for sel in sels}

cuts['Baseline_miss']['cut5'] = 'cosTheta_miss < 0.98'
cuts['Baseline_sep']['cut5']  = '((visibleEnergy > 100) | (visibleEnergy < 100 & cosTheta_miss < 0.99))'

cuts_label['Baseline_miss']['cut5'] = 'cos#theta_{miss} < 0.98'
cuts_label['Baseline_sep']['cut5']  = 'cos#theta_{miss} < 0.99 [inv]'

cuts['Baseline_tight']['cut4'] = 'zll_p > 20 & zll_p < 58'
cuts_label['Baseline_tight']['cut4'] = '20 < p_{#ell^{+}#ell^{-}} < 58 GeV'

# If a new variables is used, don't forget to add it
variables = [
    "leading_p",    "leading_theta", 
    "subleading_p", "subleading_theta",
    "zll_m", "zll_p", "zll_theta", "zll_recoil_m",
    "acolinearity",  "acoplanarity",  "zll_deltaR",
    "visibleEnergy", "cosTheta_miss", "missingMass",
    "H", "BDTscore"
]



def run(cats: list[str], 
        sels: list[str], 
        processes: dict[str, list[str]], 
        colors: dict[str, dict[str, str]], 
        legend: dict[str, dict[str, str]]
        ) -> None:
    for cat in cats:
        print(f'\n----->[Info] Making plots for {cat} channel\n')
        # first must be signal
        procs = [f"Z{cat}H" if not arg.tot else "ZH", "WW", "ZZ", "Zgamma", "Rare"]
        procs_decays = [f"z{cat}h" if not arg.tot else "zh", "WW", "ZZ", "Zgamma", "Rare"]

        inDir = get_loc(loc.EVENTS, cat, ecm, '')
        outDir   = get_loc(loc.PLOTS_MEASUREMENT, cat, ecm, '')

        branches = branches_from_cuts(cuts, variables)
        br_str = ', '.join(br for br in branches)
        print(f'----->[Info] Only importing these branches from the .root file: \n\t{br_str}\n')
        get_cutflow(inDir, outDir, 
                    cat, sels, 
                    procs, procs_decays, 
                    processes, colors, legend, 
                    cuts, cuts_label, 
                    z_decays, H_decays, 
                    branches=branches,
                    sig_scale=1, 
                    tot=arg.tot, 
                    loc_json=loc.JSON+f'/{cat}')

if __name__=='__main__':
    run(cats, sels, mk_processes(ecm=ecm), colors, labels)
    timer(t)
