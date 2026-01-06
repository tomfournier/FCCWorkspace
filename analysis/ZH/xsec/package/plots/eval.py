'''Plotting utilities for evaluating BDT performance and training diagnostics.

Provides:
- Training diagnostics visualization: `log_loss()`, `classification_error()`, `AUC()`.
- ROC curve and efficiency analysis: `roc()`, `efficiency()`, `plot_roc_curve()`.
- BDT score distributions: `bdt_score()`, `mva_score()`.
- Feature importance visualization: `importance()`.
- Significance optimization: `significance()`.
- Input variable validation: `hist_check()`.
- BDT tree structure visualization: `tree_plot()`.
- Internal helper for metric plotting: `_plot_metric()`.

Functions:
- `log_loss()`, `classification_error()`, `AUC()`: Plot training metrics across boosting iterations.
- `roc()`: Plot ROC curves for training and validation samples with AUC computation.
- `bdt_score()`: Visualize BDT score distributions for signal/background and train/validation splits.
- `mva_score()`: Plot stacked BDT scores for multiple physics processes.
- `importance()`: Display feature importance using XGBoost F-score (split count).
- `significance()`: Scan BDT score thresholds to find optimal cut for maximal significance.
- `efficiency()`: Compute and plot signal/background efficiency vs. BDT cut.
- `tree_plot()`: Render subset of boosted decision trees to Graphviz format.
- `hist_check()`: Validate input variables with per-mode histograms (train/validation split).
- `plot_roc_curve()`: Draw ROC segment with configurable threshold masking.

Conventions:
- Uses matplotlib for all plot generation with consistent styling via `set_plt_style()`.
- Supports both weighted and unweighted histograms; weights applied when available.
- ROC and significance plots mark optimal points with distinctive markers.
- Feature importance sorted in ascending order for horizontal bar charts.
- Efficiency computed as fraction of events passing BDT score threshold.
- Tree visualization uses Graphviz with evenly-spaced tree selection for large models.

Usage:
- Evaluate BDT training convergence and identify overfitting via training/validation curves.
- Optimize BDT cut threshold by scanning significance across score range.
- Validate that input variables show expected signal/background separations.
- Inspect decision tree structures for interpretability and physics alignment.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import os

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from tqdm import tqdm
from sklearn.metrics import auc

from .python.plotter import (
    set_plt_style, 
    set_labels, 
    savefigs
)
from ..tools.utils import (
    mkdir, Z,
    Significance 
)



########################
### CONFIG AND SETUP ###
########################

# Set the font, size, etc. of the label, tick, legend, etc.
set_plt_style()



########################
### HELPER FUNCTIONS ###
########################

#___________________________________________________________
def _plot_metric(results: dict[str, dict[str, list[float]]], 
                 x_axis: np.ndarray,
                 metric_name: str,
                 ylabel: str,
                 label: str, 
                 outDir: str, 
                 best_iteration: int, 
                 locx: str = 'right', 
                 locy: str = 'top', 
                 suffix: str = '', 
                 format: list[str] = ['png']) -> None:
    '''Plot a training and validation metric over boosting rounds.

    Args:
        results (dict[str, dict[str, list[float]]]): XGBoost evals_result dictionary keyed by dataset and metric.
        x_axis (np.ndarray): Boosting iteration numbers to use on the x-axis.
        metric_name (str): Metric key to plot from the evals_result.
        ylabel (str): Y-axis label for the plot.
        label (str): Text placed on the figure (e.g., channel or tag).
        outDir (str): Directory where plots are written.
        best_iteration (int): Iteration with optimal performance to mark.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    print(f'----->[Info] Plotting {metric_name}')
    fig, ax = plt.subplots()
    try:
        ax.plot(x_axis, results['validation_0'][metric_name], 
                label='Training', linewidth=2)
        ax.plot(x_axis, results['validation_1'][metric_name], 
                label='Validation', linewidth=2)
        ax.axvline(best_iteration, color='gray', 
                label='Optimal tree number')
        ax.legend()
        
        set_labels(ax, 'Number of Trees', ylabel, 
                right=label, locx=locx, locy=locy)
        savefigs(fig, outDir, metric_name.replace(' ', '_').lower(), 
                suffix=suffix, format=format)
    finally:
        plt.close(fig)



######################
### MAIN FUNCTIONS ###
######################

#_______________________________________________________
def log_loss(results: dict[str, dict[str, list[float]]], 
             x_axis: np.ndarray, 
             label: str, 
             outDir: str, 
             best_iteration: int, 
             format: list[str] = ['png'], 
             locx: str = 'right', 
             locy: str = 'top', 
             suffix: str = '') -> None:
    '''Plot log-loss evolution during training and validation.
    
    Args:
        results (dict[str, dict[str, list[float]]]): XGBoost evals_result dictionary keyed by dataset and metric.
        x_axis (np.ndarray): Boosting iteration numbers.
        label (str): Text label placed on the figure.
        outDir (str): Output directory for plots.
        best_iteration (int): Iteration to mark with optimal performance.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
        locx (str, optional): Horizontal placement for label. Defaults to 'right'.
        locy (str, optional): Vertical placement for label. Defaults to 'top'.
        suffix (str, optional): Optional suffix for filenames. Defaults to ''.
    '''
    _plot_metric(results, x_axis, 'logloss', 'Log Loss', label, outDir,
                 best_iteration, locx, locy, suffix, format)

#____________________________________________________
def classification_error(results: dict, 
                         x_axis: np.ndarray, 
                         label: str, 
                         outDir: str, 
                         best_iteration: int, 
                         locx: str = 'right', 
                         locy: str = 'top', 
                         suffix: str = '',
                         format: list[str] = ['png']
                         ) -> None:
    '''Plot classification error progression during training and validation.
    
    Args:
        results (dict): XGBoost evals_result dictionary keyed by dataset and metric.
        x_axis (np.ndarray): Boosting iteration numbers.
        label (str): Text label placed on the figure.
        outDir (str): Output directory for plots.
        best_iteration (int): Iteration to mark with optimal performance.
        locx (str, optional): Horizontal placement for label. Defaults to 'right'.
        locy (str, optional): Vertical placement for label. Defaults to 'top'.
        suffix (str, optional): Optional suffix for filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    _plot_metric(results, x_axis, 'error', 'Classification Error', label, outDir,
                 best_iteration, locx, locy, suffix, format)

#__________________________________
def AUC(results: dict, 
        x_axis: np.ndarray, 
        label: str, 
        outDir: str, 
        best_iteration: int, 
        locx: str = 'right', 
        locy: str = 'top', 
        suffix: str = '', 
        format: list[str] = ['png']
        ) -> None:
    '''Plot AUC progression during training and validation.
    
    Args:
        results (dict): XGBoost evals_result dictionary keyed by dataset and metric.
        x_axis (np.ndarray): Boosting iteration numbers.
        label (str): Text label placed on the figure.
        outDir (str): Output directory for plots.
        best_iteration (int): Iteration to mark with optimal performance.
        locx (str, optional): Horizontal placement for label. Defaults to 'right'.
        locy (str, optional): Vertical placement for label. Defaults to 'top'.
        suffix (str, optional): Optional suffix for filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    _plot_metric(results, x_axis, 'auc', 'AUC', label, outDir,
                 best_iteration, locx, locy, suffix, format)

#____________________________________________________
def plot_roc_curve(df: pd.DataFrame, 
                   column: str, 
                   thresh: float = 0.7, 
                   ax: plt.Axes | None = None, 
                   color: str | None = None, 
                   linestyle: str = '-', 
                   label: str | None = None) -> None:
    '''Draw a ROC segment above a true-positive threshold.

    Args:
        df (pd.DataFrame): Dataframe containing scores and labels.
        column (str): Score column to evaluate.
        thresh (float, optional): Minimum TPR to display; hides low-signal region. Defaults to 0.7.
        ax (plt.Axes | None, optional): Existing matplotlib axes; created if missing. Defaults to None.
        color (str | None, optional): Line color override. Defaults to None.
        linestyle (str, optional): Line style for the curve. Defaults to '-'.
        label (str | None, optional): Legend label; defaults to the column name. Defaults to None.
    '''
    if ax is None: ax = plt.gca()
    if label is None: label = column
    fpr, tpr = metrics.roc_curve(df['isSignal'], 
                                 df[column])[0:2]
    roc_auc = auc(fpr, tpr)
    mask = tpr >= thresh
    fpr, tpr = fpr[mask], tpr[mask]
    ax.plot(fpr, tpr, label=label+f', AUC={roc_auc:.2f}', 
            color=color, linestyle=linestyle, linewidth=4)

#__________________________________
def roc(df: pd.DataFrame, 
        label: str, 
        outDir: str, 
        eps: float = 0, 
        locx: str = 'right', 
        locy: str = 'top', 
        suffix: str = '',
        format: list[str] = ['png']
        ) -> None:
    '''Plot ROC curves for training and validation samples.
    
    Args:
        df (pd.DataFrame): Dataframe with `BDTscore`, `valid`, and `isSignal` columns.
        label (str): Text label placed on the figure.
        outDir (str): Output directory for plots.
        eps (float, optional): Minimum TPR threshold to display; hides low-signal region. Defaults to 0.
        locx (str, optional): Horizontal placement for label. Defaults to 'right'.
        locy (str, optional): Vertical placement for label. Defaults to 'top'.
        suffix (str, optional): Optional suffix for filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''

    print('------>[Info] Plotting ROC')
    fig, ax = plt.subplots(figsize=(12,12))
    try:
        plot_roc_curve(df[df['valid']==True],  'BDTscore', ax=ax, 
                    label='Validation Sample', thresh=eps)
        plot_roc_curve(df[df['valid']==False], 'BDTscore', ax=ax, 
                    label='Training Sample', thresh=eps,
                    linestyle='dotted')
        plt.plot([eps, 1], [eps, 1], color='navy', lw=2, linestyle='--')
        ax.legend()

        set_labels(ax, 'False Positive Rate', 'True Positive Rate', right=label, locx=locx, locy=locy)
        savefigs(fig, outDir, 'roc', suffix=suffix, format=format)
    finally:
        plt.close(fig)

#_______________________________________________
def bdt_score(df: pd.DataFrame, 
              label: str, 
              outDir: str, 
              htype: str = 'step', 
              suffix: str = '',
              locx: str = 'right', 
              locy: str = 'top', 
              yscale: str = 'log', 
              format: list[str] = ['png'],
              nbins: int = 100, 
              range: list[float | int] = [0, 1], 
              weight: bool = True, 
              unity: bool = True, 
              verbose: bool = False) -> None:
    '''Plot BDT score distributions for train/validation signal and background.

    Args:
        df (pd.DataFrame): Dataframe with `BDTscore`, `valid`, `isSignal`, and `weights`.
        label (str): Text placed on the figure (e.g., channel or tag).
        outDir (str): Directory where plots are written.
        htype (str, optional): Matplotlib histogram type. Defaults to 'step'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        yscale (str, optional): Y-axis scale (`linear` or `log`). Defaults to 'log'.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
        nbins (int, optional): Number of histogram bins. Defaults to 100.
        range (list[float | int], optional): Score range used for binning. Defaults to [0, 1].
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
        unity (bool, optional): Normalize each histogram to unit area. Defaults to True.
        verbose (bool, optional): Print category sizes when True. Defaults to False.
    '''
    
    print('----->[Info] Plotting BDT score')
    # Define styling parameters (linestyle, color, validation flag, signal flag)
    params = {
        'Signal Training':       {'l':'-',  'c':'r', 'v': False, 's': True},
        'Background Training':   {'l':'-',  'c':'b', 'v': False, 's': False},
        'Signal Validation':     {'l':'--', 'c':'r', 'v': True,  's': True},
        'Background Validation': {'l':'--', 'c':'b', 'v': True,  's': False},
    }

    # Separate data by validation status and signal/background label
    train_sig = df[(df['valid'] == False) & (df['isSignal'] == 1)]
    train_bkg = df[(df['valid'] == False) & (df['isSignal'] != 1)]
    valid_sig = df[(df['valid'] == True)  & (df['isSignal'] == 1)]
    valid_bkg = df[(df['valid'] == True)  & (df['isSignal'] != 1)]

    data_map = {
        'Signal Training': train_sig,
        'Background Training': train_bkg,
        'Signal Validation': valid_sig,
        'Background Validation': valid_bkg
    }

    fig, ax = plt.subplots()
    try:
        for param, args in params.items():
            df1 = data_map[param]
            if verbose:
                ratio = (len(df1)/float(len(df))) * 100.0
                print(f'---------> {param:<25} {len(df1)} Ratio: {ratio:.2f}')
            
            # Apply event weights if requested
            weights = df1['weights'].values if weight else None
            ax.hist(df1['BDTscore'].values, 
                    density=unity, bins=nbins, 
                    range=range, histtype=htype, 
                    label=param, linestyle=args['l'], 
                    color=args['c'], 
                    linewidth=1.5, weights=weights)
        
        ax.legend(loc='upper right', shadow=False)
        ax.set_yscale(yscale)
        yMin, yMax = ax.get_ylim()
        yMax = yMax * (10 if yscale=='log' else 1.25)
        ax.set_xlim(*range)
        ax.set_ylim(yMin, yMax) 

        ylabel = 'Normalized to Unity' if unity else 'Events'
        set_labels(ax, 'BDT Score', ylabel, 
                right=label, locx=locx, locy=locy)
        savefigs(fig, outDir, 'bdt_score', 
                suffix=suffix, format=format)
    finally:
        plt.close(fig)

#_______________________________________________
def mva_score(df: pd.DataFrame, 
              label: str, 
              outDir: str, 
              modes: list[str],
              modes_label: dict[str, str], 
              modes_color: dict[str, str],
              locx: str = 'right', 
              locy: str = 'top', 
              suffix:str = '', 
              format: list[str] = ['png'], 
              htype: str = 'step', 
              yscale: str = 'log',
              nbins: int = 100, 
              range: list[float | int] = [0, 1],
              ncols: int = 3,
              unity: bool = True, 
              weight: bool = True) -> None:
    '''Plot stacked BDT score histograms for multiple modes.

    Args:
        df (pd.DataFrame): Dataframe with sample labels and BDT scores.
        label (str): Text placed on the figure (e.g., channel or tag).
        outDir (str): Directory where plots are written.
        modes (list[str]): Ordered list of sample names.
        modes_label (dict[str, str]): Mapping from mode to legend label.
        modes_color (dict[str, str]): Mapping from mode to color.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
        htype (str, optional): Matplotlib histogram type. Defaults to 'step'.
        yscale (str, optional): Y-axis scale (`linear` or `log`). Defaults to 'log'.
        nbins (int, optional): Number of histogram bins. Defaults to 100.
        range (list[float | int], optional): Score range used for binning. Defaults to [0, 1].
        ncols (int, optional): Number of legend columns. Defaults to 3.
        unity (bool, optional): Normalize each histogram to unit area. Defaults to True.
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
    '''
    print('------>[Info] Plotting MVA score')

    # Separate data by validation status and signal/background label
    train_sig = df[(df['valid'] == False) & (df['isSignal'] == 1)]
    valid_sig = df[(df['valid'] == True)  & (df['isSignal'] == 1)]
    train_bkg = df[(df['valid'] == False) & (df['isSignal'] != 1)]
    valid_bkg = df[(df['valid'] == True)  & (df['isSignal'] != 1)]

    # Partition modes into signal-like (contain 'H') and background categories
    sig_modes = [mode for mode in modes if 'H' in mode]
    bkg_modes = [mode for mode in modes if 'H' not in mode]

    # Training with solid lines; validation with dashed lines
    datasets = [
        (train_sig, train_bkg, '-', True),
        (valid_sig, valid_bkg, '--', False)
    ]

    fig, ax = plt.subplots()
    try:
        for sig, bkg, linestyle, show_label in datasets:
            for mode in sig_modes:
                df1 = sig[sig['sample']==mode]
                weights = df1['weights'].values if weight else None
                ax.hist(df1['BDTscore'].values, 
                        density=unity, bins=nbins, 
                        range=range, 
                        histtype=htype, 
                        color=modes_color[mode],
                        label=modes_label[mode] if show_label else None, 
                        linestyle=linestyle, 
                        linewidth=1.5, weights=weights)

            # Stack background samples for overlaid histogram
            bkg_data, bkg_weights, bkg_labels = [], [], []
            bkg_colors = []
            for mode in bkg_modes:
                df1 = bkg[bkg['sample']==mode]
                bkg_data.append(df1['BDTscore'].values)
                bkg_colors.append(modes_color[mode])
                if show_label:
                    bkg_labels.append(modes_label[mode])
                if weight:
                    bkg_weights.append(df1['weights'].values)

            # Plot stacked background histogram if data exists
            if bkg_data:
                weights_stacked = bkg_weights if weight else None
                labels_stacked = bkg_labels if show_label else None
                ax.hist(bkg_data, 
                        density=unity, 
                        bins=nbins, 
                        range=range, 
                        histtype=htype, 
                        label=labels_stacked,
                        color=bkg_colors,
                        stacked=True,
                        linestyle=linestyle, 
                        linewidth=1.5, 
                        weights=weights_stacked)

        ax.set_xlim(*range)
        ax.set_yscale(yscale)
        yMin, yMax = ax.get_ylim()
        yMax = yMax * (10 if yscale=='log' else 1.25)
        ax.set_ylim(yMin, yMax)
        ax.legend(loc='upper right', shadow=False, ncols=ncols)
        
        ylabel = 'Normalized to Unity' if unity else 'Events'
        set_labels(ax, 'BDT Score', ylabel, 
                   right=label, locx=locx, locy=locy)
        savefigs(fig, outDir, 'mva_score', 
                suffix=suffix, format=format)
    finally:
        plt.close(fig)

#____________________________________________
def importance(bdt: xgb.XGBClassifier, 
               vars: list[str], 
               latex_mapping: dict[str, str], 
               label: str, 
               outDir: str, 
               locx: str = 'right', 
               locy: str = 'top', 
               suffix: str = '', 
               format: list[str] = ['png']
               ) -> None:
    '''Plot feature importance using the XGBoost F-score (split count).

    Args:
        bdt (xgb.XGBClassifier): Trained XGBClassifier instance.
        vars (list[str]): Feature names ordered as used in training.
        latex_mapping (dict[str, str]): Mapping from feature name to display label.
        label (str): Text placed on the figure (e.g., channel or tag).
        outDir (str): Directory where plots are written.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    print('------>[Info] Plotting feature importance')

    # Extract and sort feature importances by F-score (split count)
    importance = bdt.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=False)
    sorted_indices = [int(x[0][1:]) for x in sorted_importance]
    sorted_vars   = [vars[i] for i in sorted_indices]
    sorted_values = [x[1] for x in sorted_importance]

    # Map feature names to LaTeX representations for display
    sorted_vars_latex = [latex_mapping[var] for var in sorted_vars]

    # Build DataFrame and create horizontal bar chart
    importance_df = pd.DataFrame({'Variable': sorted_vars_latex, 
                                  'Importance': sorted_values})
    
    fig, ax = plt.subplots()
    try:
        importance_df.plot(kind='barh', x='Variable', 
                        y='Importance', legend=None, ax=ax)
        set_labels(ax, 'F-score', 'Variables', 
                   right=label, locx=locx, locy=locy)
        ax.grid(False, axis='y')
        savefigs(fig, outDir, 'importance', 
                 suffix=suffix, format=format)
    finally:
        plt.close(fig)

#___________________________________________
def significance(df: pd.DataFrame, 
                 label: str, 
                 outDir: str, 
                 out_txt: str, 
                 locx: str = 'right', 
                 locy: str = 'top', 
                 column: str = 'BDTscore',
                 weight: str = 'norm_weight',
                 suffix: str = '', 
                 format: list[str] = ['png']
                 ) -> None:
    '''Scan the BDT score for maximal significance and save the cut value.

    Args:
        df (pd.DataFrame): Dataframe with scores, weights, and labels.
        label (str): Text placed on the figure (e.g., channel or tag).
        outDir (str): Directory where plots are written.
        out_txt (str): Directory where the optimal cut is stored.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        column (str, optional): Score column to scan. Defaults to 'BDTscore'.
        weight (str, optional): Weight column used in the significance calculation. Defaults to 'norm_weight'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    print('------>[Info] Plotting Significance scan')

    # Extract validation signal and background samples
    df_sig = df[(df['isSignal'] == 1) & (df['valid'] == True)]
    df_bkg = df[(df['isSignal'] == 0) & (df['valid'] == True)]

    # Compute significance across BDT score thresholds
    df_Z = Significance(df_sig, df_bkg,
                        column=column,
                        weight=weight, 
                        func=Z, nbins=100)
    max_index = df_Z['Z'].idxmax()
    max_Z = df_Z.loc[max_index, 'Z']

    print(f'max-Z: {max_Z:.3f} cut threshold: {max_index:.2f}')

    # Save optimal BDT cut threshold to file
    np.savetxt(f'{out_txt}/BDT_cut.txt', [float(max_index)])
    print(f'----->[Info] Wrote BDT cut in {out_txt}/BDT_cut.txt')

    fig, ax = plt.subplots()
    try:
        ax.scatter(df_Z.index, df_Z['Z'], marker='.')
        ax.scatter(max_index, max_Z, color='r', marker='*', s=150)
        # Mark optimal cut on plot
        ax.axvline(max_index, color='gray', alpha=0.8, 
                label=f'max-Z: {max_Z:.3f}\ncut threshold: [{max_index:.2f}]')
        ax.legend()

        set_labels(ax, 'BDT score', 'Significance', 
                   right=label, locx=locx, locy=locy)
        savefigs(fig, outDir, 'significance_scan', 
                 suffix=suffix, format=format)
    finally:
        plt.close(fig)

#________________________________________________
def efficiency(df: pd.DataFrame, 
               modes: list[str], 
               modes_label: dict[str, str], 
               modes_color: dict[str, str],
               label: str, 
               outDir: str, 
               locx: str = 'right', 
               locy: str = 'top', 
               suffix: str = '',
               format: list[str] = ['png'], 
               range: list[int | float] = [0, 1], 
               incr: float = 0.01) -> None:
    '''Compute and plot efficiency vs. BDT cut for each mode.

    Args:
        df (pd.DataFrame): Dataframe with scores and sample labels.
        modes (list[str]): List of sample names to process.
        modes_label (dict[str, str]): Mapping from mode to legend label.
        modes_color (dict[str, str]): Mapping from mode to color.
        label (str): Text placed on the figure (e.g., channel or tag).
        outDir (str): Directory where plots are written.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
        range (list[int | float], optional): Inclusive BDT score range to scan. Defaults to [0, 1].
        incr (float, optional): Step size for BDT cut thresholds. Defaults to 0.01.
    '''
    
    print('------>[Info] Calculating efficiencies for each mode')
    # Create scan grid of BDT cut thresholds
    BDT_cuts = np.arange(range[0], range[1]+incr, incr)
    df_valid = df[df['valid'] == True]

    # Compute efficiency (fraction passing cut) for each mode and threshold
    eff = {}
    for mode in tqdm(modes):
        scores = df_valid[df_valid['sample']==mode]['BDTscore'].values
        eff[mode] = np.array([np.mean(scores>=cut) for cut in BDT_cuts])

    print('------>[Info] Plotting Efficiency')
    fig, ax = plt.subplots()
    try:
        for mode in modes:
            ax.plot(BDT_cuts, eff[mode], 
                    label=modes_label[mode],
                    color=modes_color[mode])
        ax.legend(loc='best', ncols=4)
        ax.set_xlim(*range)
        ax.set_ylim(None,1.3)
            
        set_labels(ax, 'BDT score', 'Efficiency', 
                right=label, locx=locx, locy=locy)
        fig.tight_layout()
        savefigs(fig, outDir, 'efficiency', 
                suffix=suffix, format=format)
    finally:
        plt.close(fig)

#__________________________________________
def tree_plot(bdt: xgb.XGBClassifier, 
              inDir: str, 
              outDir: str, 
              epochs: int, 
              n: int, 
              format: list[str] = ['png'], 
              rankdir: str = 'LR') -> None:
    '''Render a subset of boosted trees to Graphviz files.

    Args:
        bdt (xgb.XGBClassifier): Trained XGBClassifier instance.
        inDir (str): Directory containing the feature map file.
        outDir (str): Directory where rendered trees are stored.
        epochs (int): Total number of boosting iterations.
        n (int): Number of trees to render (evenly spaced up to epochs).
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
        rankdir (str, optional): Graphviz rank direction. Defaults to 'LR'.
    '''
    import graphviz

    mkdir(f'{outDir}/feature')
    # Ensure n does not exceed total number of epochs
    if n > epochs:
        n = epochs
        print('n > epochs: resetting n to match total epochs')

    print(f'------>[Info] Plotting structure of BDT')
    # Generate evenly-spaced tree indices for visualization
    num_trees = np.linspace(0, epochs-1, n, dtype=int)
    for num_tree in tqdm(num_trees):
        # Export tree to DOT format
        dot = xgb.to_graphviz(bdt, 
                              num_trees=num_tree, 
                              fmap=f'{inDir}/feature.txt', 
                              rankdir=rankdir)
        dot.save(f'{inDir}/bdt.dot')

        # Render to requested formats using graphviz
        with open(f'{inDir}/bdt.dot') as f: 
            dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
        for pl in format:
            graph.render(f'{outDir}/tmp/BDT_{num_tree}', format=pl)
    # Move rendered images to output directory and clean up
    os.system(f'mv {outDir}/tmp/*.png {outDir}/feature/')
    os.system(f'rm -rf {outDir}/tmp')
    print(f'------>[Info] Plotted structure of BDT in {outDir}/feature folder')

#__________________________________________
def hist_check(df: pd.DataFrame, 
               label: str, 
               outDir: str, 
               modes: list[str], 
               modes_label: dict[str, str],  
               modes_color: dict[str, str],
               var: str, 
               xlabel: str, 
               nbins: int = 100,
               ncols: int = 3,
               yscale: str = 'linear',
               locx: str = 'right',
               locy: str = 'top',
               suffix: str = '',
               format: list[str] = ['png'], 
               htype: str = 'step',
               unity: bool = False, 
               weight: bool = True,
               strict: bool = True
               ) -> None:
    '''Plot per-variable histograms for signal and background modes.

    Args:
        df (pd.DataFrame): Dataframe containing the variable to plot and sample metadata.
        label (str): Text placed on the figure (e.g., channel or tag).
        outDir (str): Directory where plots are written.
        modes (list[str]): Ordered list of sample names.
        modes_label (dict[str, str]): Mapping from mode to legend label.
        modes_color (dict[str, str]): Mapping from mode to color.
        var (str): Column name to histogram.
        xlabel (str): Label for the x-axis.
        nbins (int, optional): Number of histogram bins. Defaults to 100.
        ncols (int, optional): Number of legend columns. Defaults to 3.
        yscale (str, optional): Y-axis scale (`linear` or `log`). Defaults to 'linear'.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
        htype (str, optional): Matplotlib histogram type. Defaults to 'step'.
        unity (bool, optional): Normalize each histogram to unit area. Defaults to False.
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
        strict (bool, optional): If True, clamp x-limits to observed data range. Defaults to True.
    '''
    
    out = f'{outDir}/variables'
    mkdir(out)

    # Separate data by validation status and signal/background label
    train_sig = df[(df['valid'] == False) & (df['isSignal'] == 1)]
    valid_sig = df[(df['valid'] == True)  & (df['isSignal'] == 1)]
    train_bkg = df[(df['valid'] == False) & (df['isSignal'] != 1)]
    valid_bkg = df[(df['valid'] == True)  & (df['isSignal'] != 1)]

    # Partition modes into signal-like (contain 'H') and background categories
    sig_modes = [mode for mode in modes if 'H' in mode]
    bkg_modes = [mode for mode in modes if 'H' not in mode]

    # Training with solid lines; validation with dashed lines
    datasets = [
        (train_sig, train_bkg, '-', True),
        (valid_sig, valid_bkg, '--', False)
    ]

    # Determine histogram range from observed data
    xmin, xmax = df[var].min(), df[var].max()

    fig, ax = plt.subplots()
    try:
        for sig, bkg, linestyle, show_label in datasets:
            for mode in sig_modes:
                df1 = sig[sig['sample']==mode]
                weights = df1['weights'].values if weight else None
                ax.hist(df1[var].values, 
                        density=unity, 
                        bins=nbins, 
                        range=[xmin, xmax], 
                        histtype=htype, 
                        label=modes_label[mode] if show_label else None, 
                        color=modes_color[mode],
                        linestyle=linestyle, 
                        linewidth=1.5, weights=weights)

            # Stack background samples for overlaid histogram
            bkg_data, bkg_weights = [], []
            bkg_labels, bkg_colors = [], []
            for mode in bkg_modes:
                df1 = bkg[bkg['sample']==mode]
                bkg_data.append(df1[var].values)
                bkg_colors.append(modes_color[mode])
                if show_label:
                    bkg_labels.append(modes_label[mode])
                if weight:
                    bkg_weights.append(df1['weights'].values)

            # Plot stacked background histogram if data exists
            if bkg_data:
                weights_stacked = bkg_weights if weight else None
                labels_stacked = bkg_labels if show_label else None
                ax.hist(bkg_data, 
                        density=unity, 
                        bins=nbins, 
                        range=[xmin, xmax], 
                        histtype=htype, 
                        label=labels_stacked,
                        color=bkg_colors,
                        stacked=True,
                        linestyle=linestyle, 
                        linewidth=1.5, 
                        weights=weights_stacked)
        ax.legend(loc='upper center', shadow=False, ncols=ncols)

        # Constrain x-limits to data range if strict mode enabled
        if strict:
            ax.set_xlim(xmin, xmax)
        ax.set_yscale(yscale)
        yMin, yMax = ax.get_ylim()
        yMax = yMax * (10 if yscale=='log' else 1.25)
        ax.set_ylim(yMin, yMax)
        
        ylabel = 'Normalized to Unity' if unity else 'Events'
        set_labels(ax, xlabel, ylabel, 
                   right=label, locx=locx, locy=locy)
        savefigs(fig, out, var, 
                 suffix=suffix, format=format)
    finally:
        plt.close(fig)
