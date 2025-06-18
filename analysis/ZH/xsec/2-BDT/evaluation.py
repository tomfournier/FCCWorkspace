import importlib, time, argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
arg = parser.parse_args()

if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)

import tools.utils as ut

from tools.plotting import AUC, bdt_score, efficiency, roc, importance
from tools.plotting import log_loss, classification_error, significance
from tools.plotting import load_data, load_model, evaluate_bdt
from tools.plotting import print_input_summary, get_metrics

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select
from userConfig import plot_file, Label, train_vars, latex_mapping

final_state, ecm = arg.cat, arg.ecm
sel = select(arg.recoil120, arg.miss, arg.bdt)

inputDir = get_loc(loc.MVA_PROCESSED, final_state, ecm, sel)
inputBDT = get_loc(loc.BDT,           final_state, ecm, sel)
outDir   = get_loc(loc.PLOTS_BDT,     final_state, ecm, sel)

# Decay modes used in first stage training and their respective file names
ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if final_state=='ee' else f"wzp6_ee_mumu_ecm{ecm}"
modes = {f"{final_state}H":       f"wzp6_ee_{final_state}H_ecm{ecm}",
         f"ZZ":                   f"p8_ee_ZZ_ecm{ecm}", 
         f'Z{final_state}':       ee_ll,
         f"WW{final_state}":      f"p8_ee_WW_{final_state}_ecm{ecm}",
         f"egamma_{final_state}": f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
         f"gammae_{final_state}": f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
         f"gammae_{final_state}": f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
         f"egamma_{final_state}": f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}",
         f"gaga_{final_state}":   f"wzp6_gaga_{final_state}_60_ecm{ecm}"
}

vars_list = train_vars.copy()
if arg.bdt: vars_list.append("cosTheta_miss")

def plot_metrics(df, bdt, vars_list, results, 
                 x_axis, mode_names, final_state, outDir):    
    if final_state == "mumu":
      label = r"$Z(\mu^+\mu^-)H$"
    elif final_state == "ee":
      label = r"$Z(e^+e^-)H$"
    else:
      exit("ERROR: Invalid final state")
    ut.create_dir(f"{outDir}")
    log_loss(results, x_axis, label, outDir, plot_file)
    classification_error(results, x_axis, label, outDir, plot_file)
    AUC(results, x_axis, label, outDir, plot_file)
    roc(df, label, outDir, plot_file)
    bdt_score(df, label, outDir, plot_file)
    importance(bdt, vars_list, latex_mapping, label, outDir, plot_file)
    significance(df, label, outDir, inputBDT, plot_file)
    efficiency(df, mode_names, Label, label, outDir, plot_file)

df = load_data(inputDir)
print_input_summary(df, modes)
bdt = load_model(inputBDT)
df = evaluate_bdt(df, bdt, vars_list)
results, epochs, x_axis, best_iteration = get_metrics(bdt)
plot_metrics(df, bdt, vars_list, results, 
             x_axis, modes, final_state, outDir)

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
