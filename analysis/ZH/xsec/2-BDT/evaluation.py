import time, argparse, importlib

import numpy as np
import pandas as pd
import xgboost as xgb

t = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, add_package_path, get_loc, plot_file
add_package_path(loc.PACKAGE)

from package.config import timer, warning
from package.config import vars, modes_label, modes_color, vars_label, vars_xlabel

if arg.cat=='':
    warning(log_msg='Final state was not selected, please select one to run this script')

from package.tools.utils import mkdir, load_data

from package.func.bdt import load_model, get_metrics
from package.func.bdt import print_stats, evaluate_bdt

from package.plots.eval import log_loss, AUC, classification_error
from package.plots.eval import roc, bdt_score, mva_score, tree_plot
from package.plots.eval import importance, significance, efficiency
from package.plots.eval import hist_check


cat, ecm = arg.cat, arg.ecm
selections = [
    'Baseline'
]

# Decay modes used in first stage training and their respective file names
modes = {
  f'Z{cat}H':       f'wzp6_ee_{cat}H_ecm{ecm}',
  f'WW{cat}':      f'p8_ee_WW_{cat}_ecm{ecm}',
  f'ZZ':           f'p8_ee_ZZ_ecm{ecm}', 
  f'Z{cat}':       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee' 
                   else f'wzp6_ee_mumu_ecm{ecm}',
  f'egamma_{cat}': f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
  f'gammae_{cat}': f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
  f'gaga_{cat}':   f'wzp6_gaga_{cat}_60_ecm{ecm}'
}



def plot_metrics(df: pd.DataFrame, 
                 bdt: xgb.XGBClassifier, 
                 vars: list[str], 
                 results: dict[str, 
                               dict[str, 
                                    list[float]]], 
                 x_axis: np.ndarray, 
                 modes: list[str], 
                 cat: str, 
                 outDir: str) -> None: 
       
    if cat == 'mumu': label = r'$Z(\mu^+\mu^-)H$'
    elif cat == 'ee': label = r'$Z(e^+e^-)H$'
    else: warning('Invalid final state') 

    mkdir(outDir)
    log_loss(results, x_axis, label, outDir, best_iteration, format=plot_file)
    classification_error(results, x_axis, label, outDir, best_iteration, format=plot_file)
    AUC(results, x_axis, label, outDir, best_iteration, format=plot_file)
    roc(df, label, outDir, format=plot_file)
    bdt_score(df, label, outDir, format=plot_file, unity=False, nbins=500)
    mva_score(df, label, outDir, modes, modes_label, modes_color, format=plot_file, unity=False, nbins=500)
    importance(bdt, vars, vars_label, label, outDir, format=plot_file)
    significance(df, label, outDir, inBDT, format=plot_file)
    efficiency(df, modes, modes_label, modes_color, label, outDir, incr=1e-3, format=plot_file)
    tree_plot(bdt, inBDT, outDir, epochs, 20, format=plot_file)

    for var in vars:
        print(f'------>Plotting histogram for {var}')
        hist_check(df, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                   yscale='linear', suffix='_lin', format=plot_file, strict=True)
        hist_check(df, label, outDir, modes, modes_label, modes_color, var, vars_xlabel[var], 
                   yscale='log', suffix='_log', format=plot_file, strict=True)



if __name__=='__main__':
    for sel in selections:
        inDir  = get_loc(loc.MVA_INPUTS,  cat, ecm, sel)
        outDir = get_loc(loc.PLOTS_BDT,   cat, ecm, sel)
        inBDT  = get_loc(loc.BDT,         cat, ecm, sel)
        data_path = get_loc(loc.HIST_MVA, cat, ecm, sel)

        df = load_data(inDir)
        print_stats(df, modes)
        bdt = load_model(inBDT)
        df = evaluate_bdt(df, bdt, vars)
        results, epochs, x_axis, best_iteration = get_metrics(bdt)
        plot_metrics(df, bdt, vars, results, x_axis, modes, cat, outDir)

    timer(t)
