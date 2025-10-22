import importlib, time, argparse
import numpy as np
from tqdm import tqdm

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)

import tools.utils as ut

from tools.plotting import AUC, bdt_score, efficiency, roc, importance
from tools.plotting import log_loss, classification_error, significance
from tools.plotting import load_data, load_model, evaluate_bdt, tree_plot
from tools.plotting import print_input_summary, get_metrics, mva_score

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, selections
from userConfig import plot_file, Label, train_vars, latex_mapping

cat, ecm = arg.cat, arg.ecm

# Decay modes used in first stage training and their respective file names
ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if cat=='ee' else f"wzp6_ee_mumu_ecm{ecm}"
modes = {
  f"{cat}H":       f"wzp6_ee_{cat}H_ecm{ecm}",
  f"ZZ":           f"p8_ee_ZZ_ecm{ecm}", 
  f'Z{cat}':       ee_ll,
  f"WW{cat}":      f"p8_ee_WW_{cat}_ecm{ecm}",
  f"egamma_{cat}": f"wzp6_egamma_eZ_Z{cat}_ecm{ecm}",
  f"gammae_{cat}": f"wzp6_gammae_eZ_Z{cat}_ecm{ecm}",
  f"gammae_{cat}": f"wzp6_gammae_eZ_Z{cat}_ecm{ecm}",
  f"egamma_{cat}": f"wzp6_egamma_eZ_Z{cat}_ecm{ecm}",
  f"gaga_{cat}":   f"wzp6_gaga_{cat}_60_ecm{ecm}"
}

modes_color = {
  f"{cat}H":       "tab:blue",
  f"ZZ":           "tab:orange", 
  f'Z{cat}':       "tab:green",
  f"WW{cat}":      "tab:red",
  f"egamma_{cat}": "tab:purple",
  f"gammae_{cat}": "tab:brown",
  f"gammae_{cat}": "tab:pink",
  f"egamma_{cat}": "tab:gray",
  f"gaga_{cat}":   "tab:olive"
}

vars_list = train_vars.copy()

def plot_metrics(df, bdt, vars_list, results, x_axis, mode_names, cat, outDir):    
    if cat == "mumu": label = r"$Z(\mu^+\mu^-)H$"
    elif cat == "ee": label = r"$Z(e^+e^-)H$"
    else: exit("ERROR: Invalid final state")

    ut.create_dir(f"{outDir}")
    log_loss(results, x_axis, label, outDir, plot_file)
    classification_error(results, x_axis, label, outDir, plot_file)
    AUC(results, x_axis, label, outDir, plot_file)
    roc(df, label, outDir, plot_file)
    bdt_score(df, label, outDir, modes, plot_file, data_path, vars_list, unity=False, Bins=100)
    mva_score(df, label, outDir, mode_names, modes_color, Label, plot_file, data_path, vars_list, unity=False, Bins=100)
    importance(bdt, vars_list, latex_mapping, label, outDir, plot_file)
    significance(df, label, outDir, inputBDT, plot_file)
    efficiency(df, mode_names, Label, label, outDir, plot_file)
    tree_plot(bdt, inputBDT, outDir, epochs, 20, plot_file)

for sel in selections:
    inputDir  = get_loc(loc.MVA_INPUTS, cat, ecm, sel)
    inputBDT  = get_loc(loc.BDT,        cat, ecm, sel)
    outDir    = get_loc(loc.PLOTS_BDT,  cat, ecm, sel)
    data_path = get_loc(loc.MVA_INPUTS, cat, ecm, sel)

    df = load_data(inputDir)
    # print_input_summary(df, modes)
    bdt = load_model(inputBDT)
    df = evaluate_bdt(df, bdt, vars_list)
    results, epochs, x_axis, best_iteration = get_metrics(bdt)
    plot_metrics(df, bdt, vars_list, results, x_axis, modes, cat, outDir)

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
