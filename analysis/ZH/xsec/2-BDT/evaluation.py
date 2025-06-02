import importlib, time

t1 = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, train_vars, mode_names, latex_mapping
from userConfig import final_state, plot_file, Label, miss_BDT

import tools.utils as ut

from tools.plotting import AUC, bdt_score, efficiency, roc, importance
from tools.plotting import log_loss, classification_error, significance
from tools.plotting import load_data, load_model, evaluate_bdt
from tools.plotting import print_input_summary, get_metrics

inputDir = loc.MVA_PROCESSED
inputBDT = loc.BDT
outDir = loc.PLOTS_BDT

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
print_input_summary(df, mode_names)
bdt = load_model(inputBDT)
df = evaluate_bdt(df, bdt, train_vars)
results, epochs, x_axis, best_iteration = get_metrics(bdt)
plot_metrics(df, bdt, train_vars, results, 
             x_axis, mode_names, final_state, outDir)

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
