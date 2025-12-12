import time, argparse,importlib
t = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat',  help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='ee-mumu')
parser.add_argument('--ecm',  help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', 
                    choices=[10.8, 3.1], type=float, default=10.8)

parser.add_argument('--yields',  help='Do not make yields plots',            action='store_true')
parser.add_argument('--decay',   help='Do not make Higgs decays only plots', action='store_true')
parser.add_argument('--make',    help='Do not make distribution plots',      action='store_true')
parser.add_argument('--scan',    help='Make significance scan plots',        action='store_true')

parser.add_argument('--tot',    help='Include all the Z decays in the plots', action='store_true')
arg = parser.parse_args()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc

from package.config import timer, mk_processes, z_decays, h_decays, H_decays, colors, labels
from package.plots.plotting import get_args, args_decay, significance, makePlot, PlotDecays, AAAyields

cats, ecm, lumi = arg.cat.split('-'), arg.ecm, arg.lumi

sels = [
    'Baseline'#, 'Baseline_high', 'Baseline_low',
    # 'Baseline_miss', 'Baseline_miss_high', 'Baseline_miss_low', 
    
    # 'Baseline_vis', 'Baseline_inv',
    # 'Baseline_sep', 'Baseline_sep_high', 'Baseline_sep_low',
    # 'Baseline_tight', 'Baseline_tight_high', 'Baseline_tight_low'
]
processes = mk_processes(ecm=ecm)
variables = [
    # Leptons
    'leading_p', 'leading_theta', 'subleading_p', 'subleading_theta',
    # 'leading_phi', 'subleading_phi',
    # Zed
    'zll_m', 'zll_p', 'zll_theta', # 'zll_phi',
    # more control variables
    'acolinearity', 'acoplanarity', 'zll_deltaR',
    # Recoil
    'zll_recoil_m',
    # missing Information
    'visibleEnergy', 'cosTheta_miss', 'missingMass',
    # Higgsstrahlungness
    'H',
    # BDT score
    'BDTscore'
]

plots = {}
plots['signal'] = {
    'ZH':[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays]
}
plots['backgrounds'] = {
    'WW':    [f'p8_ee_WW_ecm{ecm}', f'p8_ee_WW_ee_ecm{ecm}', f'p8_ee_WW_mumu_ecm{ecm}'],
    'ZZ':    [f'p8_ee_ZZ_ecm{ecm}'], 
    'Zgamma':[f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
              f'wzp6_ee_tautau_ecm{ecm}'],
    'Rare':  [f'wzp6_egamma_eZ_Zee_ecm{ecm}',    f'wzp6_gammae_eZ_Zee_ecm{ecm}',
              f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',  f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
              f'wzp6_gaga_ee_60_ecm{ecm}',       f'wzp6_gaga_mumu_60_ecm{ecm}',
              f'wzp6_gaga_tautau_ecm{ecm}',      f'wzp6_ee_nuenueZ_ecm{ecm}']
}

args = {
    'cosTheta_miss': {'xmin':0.9},
    'BDTscore': {'ymin':1, 'ymax':1e5, 'rebin':2, 'which':'make'},
    'zll_recoil_m': {'xmin':120, 'xmax':140, 'sel':'*_high'}
}    



def run(cats, sels, vars, processes, colors, legend):
    for cat in cats:
        print(f'\n----->[Info] Making plots for {cat} channel\n')
        # first must be signal
        procs = ['', 'WW', 'ZZ', 'Zgamma', 'Rare']
        procs[0] = f'Z{cat}H' if not arg.tot else 'ZH'

        inDir  = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')
        outDir = get_loc(loc.PLOTS_MEASUREMENT, cat, ecm, '')

        for sel in sels:
            if not arg.yields and not arg.make and not arg.decay and not arg.scan:
                print(f'\n----->[Info] Making plots for {sel} selection\n')

            if not arg.yields: 
                AAAyields('zll_p', inDir, outDir, plots, legend, colors, cat, sel)
            if not arg.make or not arg.decay or arg.scan:
                for var in vars:
                    print(f'\n----->[Info] Making plots for {var}')

                    if arg.scan:
                        significance(var, inDir, outDir, sel, procs, processes, reverse=True)
                        significance(var, inDir, outDir, sel, procs, processes, reverse=False)

                    if not arg.decay: 
                        kwarg_decay = args_decay(var, sel, args)
                        PlotDecays(var, inDir, outDir, sel, [cat], H_decays, logY=False, tot=False, **kwarg_decay)
                        PlotDecays(var, inDir, outDir, sel, [cat], H_decays, logY=True,  tot=False, **kwarg_decay)

                        PlotDecays(var, inDir, outDir, sel, z_decays, H_decays, logY=False, tot=True, **kwarg_decay)
                        PlotDecays(var, inDir, outDir, sel, z_decays, H_decays, logY=True,  tot=True, **kwarg_decay)
                    if not arg.make:  
                        kwarg = get_args(var, sel, args)
                        makePlot(var, inDir, outDir, sel, procs, processes, colors, legend, logY=False, **kwarg)
                        makePlot(var, inDir, outDir, sel, procs, processes, colors, legend, logY=True,  **kwarg)

if __name__=='__main__':
    run(cats, sels, variables, processes, colors, labels)
    timer(t)
