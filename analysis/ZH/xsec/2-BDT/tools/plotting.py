import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import awkward as ak
import matplotlib as mpl
from tqdm import tqdm
from matplotlib import rc
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
from . import utils as ut

rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)

#__________________________________________________________
def errorbar_hist(P, var, P_name, title, units, low, high, bins, outDir):
    fig, ax = plt.subplots(figsize=(8,8))
    #Number of events, use this to determine bins and thus bin width
    n = np.sum(ak.num(P).tolist())
    bin_w = (high - low)/bins

    counts, bin_edges = np.histogram(ak.to_list(ak.flatten(P[var])), bins, range=(low,high))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    err = np.sqrt(counts)
    plt.errorbar(bin_centres, counts, yerr=err, fmt='o', color='k')
    plt.xlabel(f"{title} [{units}]",fontsize=30)
    plt.ylabel("Candidates / (%.4f %s)" % (bin_w, units), fontsize=30)
    plt.xlim(low,high)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ymin, ymax = plt.ylim()
    plt.ylim(0.,ymax*1.1)
    plt.tight_layout()
    fig.savefig(f"{outDir}/{P_name}_{var}.pdf")

#__________________________________________________________
def errorbar_plot(x_vals, y_vals, x_name, y_name, x_title, y_title, 
                  x_range, y_range, outDir, x_err=None, y_err=None):
    fig, ax = plt.subplots(figsize=(8,8))
    plt.errorbar(x_vals, y_vals, xerr=x_err, yerr=y_err, fmt='o', color='k')
    plt.xlabel(x_title,fontsize=30)
    plt.xlim(x_range[0],x_range[1])
    plt.ylabel(y_title,fontsize=30)
    plt.ylim(y_range[0],y_range[1])
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    fig.savefig(f"{outDir}/{x_name}_vs_{y_name}.pdf")

#__________________________________________________________
def hist_plot(X, X_name, title, low, high, bins, outDir):
    fig, ax = plt.subplots(figsize=(8,8))
    plt.hist(X,bins=bins,range=(low,high),histtype='step',color='k')
    plt.xlabel(title,fontsize=30)
    plt.xlim(low,high)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ymin, ymax = plt.ylim()
    plt.ylim(0.,ymax*1.1)
    plt.tight_layout()
    fig.savefig(f"{outDir}/{X_name}.pdf")

#__________________________________________________________
def hist_plot_2d(X, X_name, X_title, Y, Y_name, Y_title, 
                 X_low, X_high, Y_low, Y_high, X_bins, Y_bins, 
                 log, outDir):
    fig, ax = plt.subplots(figsize=(8,8))
    if(log==True):
        norm_opt = mpl.colors.LogNorm()
    else:
        norm_opt = mpl.colors.Normalize()
    plt.hist2d(X.tolist(),Y.tolist(),bins=[X_bins,Y_bins],range=[[X_low, X_high], [Y_low, Y_high]], norm=norm_opt)
    plt.xlabel(X_title,fontsize=30)
    plt.xlim(X_low,X_high)
    plt.ylabel(Y_title,fontsize=30)
    plt.ylim(Y_low,Y_high)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    fig.savefig(f"{outDir}/{X_name}_vs_{Y_name}.pdf")

#__________________________________________________________
def roc_curve(df, score_column, tpr_threshold=0.7, ax=None, color=None, linestyle='-', label=None):
  if ax is None:
    ax = plt.gca()
  if label is None:
    label = score_column
  fpr, tpr, thresholds = roc_curve(df['isSignal'], df[score_column], sample_weight=df['weight_total'])
  roc_auc = auc(fpr, tpr)
  mask = tpr >= tpr_threshold
  fpr, tpr = fpr[mask], tpr[mask]
  ax.plot(fpr, tpr, label=label+', AUC = {:.2f}'.format(roc_auc), color=color, linestyle=linestyle)

#__________________________________________________________
def plot_roc_curve(df, score_column, tpr_threshold=0.7, ax=None, color=None, linestyle='-', label=None):
    if ax is None:
        ax = plt.gca()
    if label is None:
        label = score_column
    fpr, tpr, thresholds = metrics.roc_curve(df['isSignal'], df[score_column] )
    roc_auc = auc(fpr, tpr)
    mask = tpr >= tpr_threshold
    fpr, tpr = fpr[mask], tpr[mask]
    ax.plot(fpr, tpr, label=label+', AUC={:.2f}'.format(roc_auc), color=color, linestyle=linestyle, linewidth=4)

#__________________________________________________________
def load_data(inputDir):
    path = f"{inputDir}"
    df = pd.read_pickle(f"{path}/preprocessed.pkl")
    return df

#__________________________________________________________
def print_input_summary(df, modes):
    print(f"__________________________________________________________")
    print(f"Input number of events:")
    for cur_mode in modes:
        print(f"Number of training {cur_mode}: {int(len(df[(df['sample'] == cur_mode) & (df['valid'] == False)]))}")
        print(f"Number of validation {cur_mode}: {int(len(df[(df['sample'] == cur_mode) & (df['valid'] == True)]))}")
    print(f"__________________________________________________________")

#__________________________________________________________
def load_model(inputDir):
    print(f"--->Loading BDT model {inputDir}/xgb_bdt.joblib")
    bdt = joblib.load(f"{inputDir}/xgb_bdt.joblib")
    return bdt

#__________________________________________________________
def evaluate_bdt(df, bdt, vars_list):
    X = df[vars_list]
    print(f"--->Evaluating BDT model")
    df["BDTscore"] = bdt.predict_proba(X).tolist()
    df["BDTscore"] = df["BDTscore"].apply(lambda x: x[1])
    return df

#__________________________________________________________
def get_metrics(bdt):
    print("------>Retrieving performance metrics")
    results = bdt.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    best_iteration = bdt.best_iteration + 1
    return results, epochs, x_axis, best_iteration

#__________________________________________________________
def log_loss(results, x_axis,label, outDir, plot_file):
    print("------>Plotting log loss")
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Training')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
    #plt.axvline(best_iteration, color="gray", label="Optimal tree number")
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel("Number of trees", fontsize=30)
    plt.ylabel('Log Loss', fontsize=30)
    ax.set_title(r'$\textbf{\textit{FCC-ee Simulation}}$', fontsize=20, loc='left')
    ax.set_title(label, fontsize=20, loc='right')
    plt.savefig(f"{outDir}/log_loss.{plot_file}", bbox_inches='tight')
    print(f"Saved log loss plot to {outDir}/log_loss.{plot_file}")
    plt.close()

#__________________________________________________________
def classification_error(results, x_axis, label, outDir, plot_file):
    print("------>Plotting classification error")
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x_axis, results['validation_0']['error'], label='Training')
    ax.plot(x_axis, results['validation_1']['error'], label='Validation')
    #plt.axvline(best_iteration, color="gray", label="Optimal tree number")
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel('Number of trees', fontsize=30)
    plt.ylabel('Classification Error', fontsize=30)
    ax.set_title(r'$\textbf{\textit{FCC-ee Simulation}}$', fontsize=20, loc='left')
    ax.set_title(label, fontsize=20, loc='right')
    plt.savefig(f"{outDir}/classification_error.{plot_file}", bbox_inches='tight')
    print(f"Saved classification error plot to {outDir}/classification_error.{plot_file}")
    plt.close()

#__________________________________________________________
def AUC(results, x_axis, label, outDir, plot_file):
    print("------>Plotting AUC")
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x_axis, results['validation_0']['auc'], label='Training', linewidth=2)
    ax.plot(x_axis, results['validation_1']['auc'], label='Validation', linewidth=2)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel('Number of trees', fontsize=30)
    plt.ylabel('AUC', fontsize=30)
    ax.set_title(r'$\textbf{\textit{FCC-ee Simulation}}$', fontsize=20, loc='left')
    ax.set_title(label, fontsize=20, loc='right')
    plt.savefig(f"{outDir}/auc.{plot_file}", bbox_inches='tight')
    print(f"Saved AUC plot to {outDir}/auc.{plot_file}")
    plt.close()

#__________________________________________________________
def roc(df, label, outDir, plot_file):

    print("------>Plotting ROC")
    fig, axes = plt.subplots(1, 1, figsize=(12,12))
    eps=0.
    ax=axes
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel("False positive rate", fontsize=30)
    ax.set_ylabel("True positive rate", fontsize=30)
    plot_roc_curve(df[df['valid']==True],  "BDTscore", ax=ax, 
                   label="Validation Sample", tpr_threshold=eps)
    plot_roc_curve(df[df['valid']==False], "BDTscore", ax=ax, 
                   label="Training Sample", color="#ff7f02", tpr_threshold=eps,
                   linestyle='dotted')
    plt.plot([eps, 1], [eps, 1], color='navy', lw=2, linestyle='--')
    ax.legend(fontsize=14)
    ax.set_title(r'$\textbf{\textit{FCC-ee Simulation}}$', fontsize=20, loc='left')
    ax.set_title(label, fontsize=20, loc='right')
    fig.savefig(f"{outDir}/roc.{plot_file}", bbox_inches='tight')
    print(f"Saved ROC plot to {outDir}/roc.{plot_file}")
    plt.close()

#__________________________________________________________
def bdt_score(df, label, outDir, plot_file):
    print("------>Plotting BDT score (overtraining check)")
    
    fig, ax = plt.subplots(figsize=(12,8))
    Bins = 20
    htype = "step"
    
    tag = ['Signal Training', 'Signal Validation', 'Background Training', 'Background Validation']
    line = ['solid', 'dashed', 'solid', 'dashed']
    color = ['red', 'red', 'blue', 'blue']
    cut = ['valid==False & isSignal==1', 'valid==True & isSignal==1', 
           'valid==False & isSignal!=1', 'valid==True & isSignal!=1']
    
    for (x, y, z, w) in zip(tag, line, color, cut):
        df_instance = df.query(w)
        print(f'---------> {x} {len(df_instance)} "Ratio: {((len(df_instance)/float(len(df))) * 100.0):.2f}')
        ax.hist(df_instance['BDTscore'], density=True, bins=Bins, range=[0.0, 1.0], histtype=htype, 
                label=x, linestyle=y, color=z, linewidth=1.5)
    
    plt.yscale('log')
    ax.legend(loc="upper right", frameon=False, shadow=False, fontsize=14)
    ax.set_title(r'$\textbf{\textit{FCC-ee Simulation}}$', fontsize=20, loc='left')
    ax.set_title(label, fontsize=20, loc='right')

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel("BDT Score", fontsize=30, loc='right', weight='bold')  
    ax.set_ylabel("Normalized to Unity", fontsize=30, loc='top', weight='bold')  
     
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    ax.set_ylim(top=ax.get_ylim()[1] * 3)  # Increase the Y-axis space
    ax.set_xlim(left=0.0, right=1.0) 

    print("------>Plotting BDT score")
    plt.savefig(f"{outDir}/bdt_score.{plot_file}", bbox_inches='tight')
    print(f"Saved BDT score to {outDir}/bdt_score.{plot_file}")
    plt.close()

#__________________________________________________________
def importance(bdt, vars_list, latex_mapping, label, outDir, plot_file):
    print("------>Plotting feature importance")
    print("------>Plotting importance")
    fig, ax = plt.subplots(figsize=(12,8))

    # Get feature importances and sort them by importance
    importance = bdt.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=False)

    # Get the sorted indices of the variables
    sorted_indices = [int(x[0][1:]) for x in sorted_importance]

    # Get the sorted variable names and their corresponding importances
    sorted_vars = [vars_list[i] for i in sorted_indices]
    sorted_values = [x[1] for x in sorted_importance]

    # Update variable names with their LaTeX versions
    sorted_vars_latex = [latex_mapping[var] for var in sorted_vars]

    # Create a DataFrame and plot the feature importances
    importance_df = pd.DataFrame({'Variable': sorted_vars_latex, 'Importance': sorted_values})
    importance_df.plot(kind='barh', x='Variable', y='Importance', legend=None, ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel('F-score', fontsize=30)
    ax.set_ylabel('Variables', fontsize=30)
    ax.set_title(r'$\textbf{\textit{FCC-ee simulation}}$', fontsize=20, loc='left')
    ax.set_title(label, fontsize=20, loc='right')
    plt.savefig(f"{outDir}/importance.{plot_file}", bbox_inches='tight')
    print(f"------>Saved Importance to {outDir}/importance.{plot_file}")
    plt.close()

#__________________________________________________________
def significance(df, label, outDir, out_txt, plot_file):
    print("------>Plotting Significance scan")
    #compute the significance
    df_Z = ut.Significance(df[(df['isSignal'] == 1) & (df['valid'] == True)], 
                           df[(df['isSignal'] == 0) & (df['valid'] == True)], 
                           score_column='BDTscore', func=ut.Z, nbins=100)
    max_index=df_Z["Z"].idxmax()
    print(f'max-Z: {df_Z.loc[max_index,"Z"]:.2f} cut threshold: [{max_index}]')

    np.savetxt(f"{out_txt}/BDT_cut.txt", [np.round(max_index, 2)])
    print(f"----->[Info] Wrote BDT cut in {out_txt}/BDT_cut.txt")

    fig, ax = plt.subplots(figsize=(12,8))
    plt.scatter(df_Z.index, df_Z["Z"], marker='.')
    ax.scatter(x=max_index, y=df_Z.loc[max_index,"Z"], c='r', marker="*")
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel("BDT Score ", fontsize=30)
    plt.ylabel("Significance", fontsize=30)
    txt1 = Rectangle((0, 0), .01, .01, fc="w", fill=False, edgecolor='none', linewidth=0)
    txt2 = Rectangle((0, 0), .01, .01, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([txt1, txt2], (f'max-Z: {df_Z.loc[max_index,"Z"]:.2f}\ncut threshold: [{max_index:.2f}]', '$Z = S/\\sqrt{S+B}$'), 
               fontsize=14, frameon=False)
    ax.set_title(r'$\textbf{\textit{FCC-ee Simulation}}$', fontsize=20, loc='left')
    ax.set_title(label, fontsize=20, loc='right')
    print("------>Plotting significance scan")
    plt.savefig(f"{outDir}/significance_scan.{plot_file}", bbox_inches='tight')
    print(f"------>Saved Importance to {outDir}/significance_scan.{plot_file}")
    plt.close()

#__________________________________________________________
def efficiency(df, mode_names, Label, label, outDir, plot_file):
    
    #Plot efficiency as a function of BDT cut in each sample
    print("------>Plotting Efficiency")
    BDT_cuts = np.linspace(0,99,99)
    cut_vals = []
    eff = {}

    for cur_mode in mode_names:
      eff[cur_mode] = []

    for x in tqdm(BDT_cuts):
      cut_val = float(x)/100
      cut_vals.append(cut_val)
      for cur_mode in mode_names:
        eff[cur_mode].append(float(len(df[(df['sample'] == cur_mode) & (df['valid'] == True) & (df['BDTscore'] > cut_val)]))/float(len(df[(df['sample'] == cur_mode) & (df['valid'] == True)])))
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    for cur_mode in mode_names:
      plt.plot(cut_vals, eff[cur_mode], label=Label[cur_mode])
         
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.xlim(0,1)
    plt.xlabel("BDT score",fontsize=30)
    plt.ylabel("Efficiency",fontsize=30)

    ymin,ymax = plt.ylim()
    plt.ylim(ymin,1.3)
    plt.legend(fontsize=14, loc="best", ncols=3)
    plt.grid(alpha=0.4,which="both")
    ax.set_title(r'$\textbf{\textit{FCC-ee Simulation}}$', fontsize=16, loc='left')
    ax.set_title(label, fontsize=16, loc='right')
    plt.tight_layout()
    print("------>Plotting efficiency")
    plt.savefig(f"{outDir}/efficiency.{plot_file}", bbox_inches='tight')
    print(f"------>Saved Efficiency to {outDir}/efficiency.{plot_file}")
    plt.close()
