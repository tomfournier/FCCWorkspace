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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from tqdm import tqdm

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xgboost as xgb

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
plt.ioff()  # Disable interactive mode for faster plotting

# Cache for splits to avoid recomputation if same DataFrame used
_splits_cache = {}
_cache_id = None



########################
### HELPER FUNCTIONS ###
########################

#____________________________
def _make_fig(
    figsize: tuple = (12, 8), 
    dpi: int = 100
    ) -> tuple[plt.Figure, 
               plt.Axes]:
    '''Create and pre-configure a matplotlib figure.
    
    Ensures consistent figure setup across all plotting functions.
    
    Args:
        figsize (tuple, optional): Figure size (width, height). Defaults to (10, 8).
        dpi (int, optional): Figure DPI. Defaults to 100.
        
    Returns:
        Tuple of (fig, ax) ready to use
    '''
    return plt.subplots(figsize=figsize, dpi=dpi)

#_____________________________
def _get_splits(
    df: 'pd.DataFrame'
    ) -> dict[str, 
              'pd.DataFrame']:
    '''Fast extraction of train/valid and signal/bkg splits.
    
    Computes boolean masks once, reused across functions.
    Caches results if same DataFrame used multiple times.
    
    Args:
        df: Input DataFrame with 'valid' and 'isSignal' columns
        
    Returns:
        dict with precomputed masks and split DataFrames
    '''
    global _splits_cache, _cache_id
    
    # Use DataFrame memory address as cache key
    df_id = id(df)
    
    # Return cached result if same DataFrame
    if df_id == _cache_id and df_id in _splits_cache:
        return _splits_cache[df_id]
    
    # Compute boolean masks explicitly to avoid integer bitwise quirks
    is_valid = df['valid'].to_numpy(dtype=bool, copy=False)
    is_sig = (df['isSignal'] == 1).to_numpy(copy=False)
    
    train_sig = ~is_valid & is_sig
    train_bkg = ~is_valid & ~is_sig
    val_sig = is_valid & is_sig
    val_bkg = is_valid & ~is_sig
    
    result = {
        'train_sig': df.loc[train_sig, :],
        'train_bkg': df.loc[train_bkg, :],
        'val_sig':   df.loc[val_sig, :],
        'val_bkg':   df.loc[val_bkg, :]
    }
    
    # Cache for future calls with same DataFrame
    _cache_id = df_id
    _splits_cache[df_id] = result
    
    # Keep cache small (max 3 DataFrames)
    if len(_splits_cache) > 3:
        _splits_cache.clear()
    
    return result

#____________________________
def _get_wts(
    df: 'pd.DataFrame', 
    use_wts: bool,
    weights: str = 'weights'
    ) -> 'np.ndarray' | None:
    '''Extract weights efficiently, return None if unused.
    
    Args:
        df: Input DataFrame
        use_wts: Whether to use weights
        
    Returns:
        Weights array or None
    '''
    return df[weights].values if use_wts else None

#____________________________________
def _plot_metric(
    results: dict[str, 
                  dict[str, 
                       list[float]]], 
    x_axis: 'np.ndarray',
    metric_name: str,
    ylabel: str,
    label: str, 
    outDir: str, 
    best_iteration: int, 
    locx: str = 'right', 
    locy: str = 'top', 
    suffix: str = '', 
    dpi: int = 100,
    format: list[str] = ['png']
    ) -> None:
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
    fig, ax = _make_fig(dpi=dpi)
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

#____________________________________
def log_loss(
    results: dict[str, 
                  dict[str, 
                       list[float]]], 
    x_axis: 'np.ndarray', 
    label: str, 
    outDir: str, 
    best_iteration: int, 
    format: list[str] = ['png'], 
    locx: str = 'right', 
    locy: str = 'top', 
    suffix: str = '',
    dpi: int = 100
    ) -> None:
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
        dpi (int, optional): Figure DPI. Defaults to 100.
    '''
    _plot_metric(results, x_axis, 'logloss', 'Log Loss', label, outDir,
                 best_iteration, locx, locy, suffix, dpi, format)

#_______________________________
def classification_error(
    results: dict, 
    x_axis: 'np.ndarray', 
    label: str, 
    outDir: str, 
    best_iteration: int, 
    locx: str = 'right', 
    locy: str = 'top', 
    suffix: str = '',
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    _plot_metric(results, x_axis, 'error', 'Classification Error', label, outDir,
                 best_iteration, locx, locy, suffix, dpi, format)

#_____________________________
def AUC(
    results: dict, 
    x_axis: np.ndarray, 
    label: str, 
    outDir: str, 
    best_iteration: int, 
    locx: str = 'right', 
    locy: str = 'top', 
    suffix: str = '', 
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    _plot_metric(results, x_axis, 'auc', 'AUC', label, outDir,
                 best_iteration, locx, locy, suffix, dpi, format)

#_____________________________________
def plot_roc_curve(
    df: 'pd.DataFrame', 
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
    import numpy as np
    import sklearn.metrics as metrics
    from sklearn.metrics import auc

    if ax is None: ax = plt.gca()
    if label is None: label = column
    is_sig = df['isSignal'].to_numpy(np.int8, copy=False)
    scores = df[column].to_numpy(np.float32, copy=False)
    fpr, tpr = metrics.roc_curve(is_sig, scores)[0:2]
    roc_auc = auc(fpr, tpr)
    mask = tpr >= thresh
    fpr, tpr = fpr[mask], tpr[mask]
    ax.plot(fpr, tpr, label=label+f', AUC={roc_auc:.2f}', 
            color=color, linestyle=linestyle, linewidth=4)

#______________________________
def roc(
    df: 'pd.DataFrame', 
    label: str, 
    outDir: str, 
    eps: float = 0, 
    locx: str = 'right', 
    locy: str = 'top', 
    suffix: str = '',
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''

    print('------>[Info] Plotting ROC')
    fig, ax = _make_fig(figsize=(12, 12), dpi=dpi)
    try:
        plot_roc_curve(df[df['valid']==True],  'BDTscore', ax=ax, 
                    label='Validation Sample', thresh=eps)
        plot_roc_curve(df[df['valid']==False], 'BDTscore', ax=ax, 
                    label='Training Sample', thresh=eps,
                    linestyle=(0, (1, 1)))
        ax.plot([eps, 1], [eps, 1], color='r', lw=2, linestyle='--', label='Random Classifier')
        ax.plot([eps, eps, 1], [eps, 1, 1], color='g', lw=2, linestyle='--', label='Perfect Classifier')
        ax.legend()

        set_labels(ax, 'False Positive Rate', 'True Positive Rate', right=label, locx=locx, locy=locy)
        savefigs(fig, outDir, 'roc', suffix=suffix, format=format)
    finally:
        plt.close(fig)

#_____________________________________
def bdt_score(
    df: 'pd.DataFrame', 
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
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        verbose (bool, optional): Print category sizes when True. Defaults to False.
    '''
    print('------>[Info] Plotting BDT score')

    # Get splits with single pass through data
    splits = _get_splits(df)
    df_len = len(df)
    
    # Define styling: linestyle, color, valid flag, sig flag
    params = [
        ('Signal Training',       '-',  'r', splits['train_sig']),
        ('Background Training',   '-',  'b', splits['train_bkg']),
        ('Signal Validation',     '--', 'r', splits['val_sig']),
        ('Background Validation', '--', 'b', splits['val_bkg']),
    ]

    fig, ax = _make_fig(dpi=dpi)
    try:
        for name, ls, c, d in params:
            if verbose:
                ratio = (len(d) / df_len) * 100.0
                print(f'---------> {name:<25} {len(d):6d} Ratio: {ratio:5.2f}%')
            
            wts = _get_wts(d, weight)
            ax.hist(d['BDTscore'].values, 
                    density=unity, bins=nbins, 
                    range=range, histtype=htype, 
                    label=name, linestyle=ls, 
                    color=c, linewidth=1.5, weights=wts)
        
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

#______________________________________
def mva_score(
    df: 'pd.DataFrame', 
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
    dpi: int = 100,
    weight: bool = True
    ) -> None:
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
    '''
    print('------>[Info] Plotting MVA score')

    # Get splits with single pass
    splits = _get_splits(df)
    sig_modes = [m for m in modes if 'H' in m]
    bkg_modes = [m for m in modes if 'H' not in m]

    fig, ax = _make_fig(dpi=dpi)
    try:
        for linestyle, show_lbl in [('-', True), ('--', False)]:
            sig_data = splits['train_sig'] if linestyle == '-' else splits['val_sig']
            bkg_data = splits['train_bkg'] if linestyle == '-' else splits['val_bkg']
            
            # Plot signal modes
            for mode in sig_modes:
                d = sig_data[sig_data['sample'] == mode]
                wts = _get_wts(d, weight)
                ax.hist(
                    d['BDTscore'].values, 
                    density=unity, 
                    bins=nbins, 
                    range=range, 
                    histtype=htype, 
                    color=modes_color[mode],
                    label=modes_label[mode] if show_lbl else None, 
                    linestyle=linestyle, 
                    linewidth=1.5, 
                    weights=wts)

            # Stack background samples
            bkg_arrays, bkg_wts, bkg_lbls, bkg_cols = [], [], [], []
            for mode in bkg_modes:
                d = bkg_data[bkg_data['sample'] == mode]
                if len(d) > 0:
                    bkg_arrays.append(d['BDTscore'].values)
                    bkg_cols.append(modes_color[mode])
                    if show_lbl:
                        bkg_lbls.append(modes_label[mode])
                    if weight:
                        bkg_wts.append(d['weights'].values)

            if bkg_arrays:
                wts_stacked = bkg_wts if weight else None
                lbls_stacked = bkg_lbls if show_lbl else None
                ax.hist(
                    bkg_arrays, 
                    density=unity, 
                    bins=nbins, 
                    range=range, 
                    histtype=htype, 
                    label=lbls_stacked, 
                    color=bkg_cols,
                    stacked=True, 
                    linestyle=linestyle, 
                    linewidth=1.5, 
                    weights=wts_stacked)

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

#_________________________________
def importance(
    bdt: 'xgb.XGBClassifier', 
    vars: list[str], 
    latex_mapping: dict[str, str], 
    label: str, 
    outDir: str, 
    locx: str = 'right', 
    locy: str = 'top', 
    suffix: str = '', 
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    print('------>[Info] Plotting feature importance')

    import pandas as pd

    # Extract and sort feature importances by F-score (split count)
    importance = bdt.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1])
    
    # Map feature indices to LaTeX labels and extract values in single pass
    sorted_vars_latex = [latex_mapping[vars[int(k[1:])]] for k, _ in sorted_importance]
    sorted_values = [v for _, v in sorted_importance]

    # Build DataFrame and create horizontal bar chart
    importance_df = pd.DataFrame(
        {'Variable': sorted_vars_latex, 
         'Importance': sorted_values}
    )
    
    fig, ax = _make_fig(dpi=dpi)
    try:
        importance_df.plot(
            kind='barh', x='Variable', y='Importance', 
            legend=None, ax=ax
        )
        set_labels(
            ax, 'F-score', 'Variables', 
            right=label, locx=locx, locy=locy
        )
        ax.grid(False, axis='y')
        savefigs(
            fig, outDir, 'importance', 
            suffix=suffix, format=format
        )
    finally:
        plt.close(fig)

#_______________________________
def significance(
    df: 'pd.DataFrame', 
    label: str, 
    outDir: str, 
    out_txt: str, 
    locx: str = 'right', 
    locy: str = 'top', 
    column: str = 'BDTscore',
    weight: str = 'norm_weight',
    suffix: str = '', 
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        format (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    import numpy as np
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

    fig, ax = _make_fig(dpi=dpi)
    try:
        ax.scatter(df_Z.index, df_Z['Z'], marker='.')
        ax.scatter(max_index, max_Z, color='r', marker='*', s=150)
        # Mark optimal cut on plot
        ax.axvline(max_index, color='gray', alpha=0.8, 
                label=f'max-Z: {max_Z:.3f}\ncut threshold: [{max_index:.2f}]')
        ax.legend()

        set_labels(
            ax, 'BDT score', 'Significance', 
            right=label, locx=locx, locy=locy
        )
        savefigs(
            fig, outDir, 'significance_scan', 
            suffix=suffix, format=format
        )
    finally:
        plt.close(fig)

#____________________________________
def efficiency(
    df: 'pd.DataFrame', 
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
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        incr (float, optional): Step size for BDT cut thresholds. Defaults to 0.01.
    '''
    import numpy as np
    print('------>[Info] Calculating efficiencies for each mode')
    
    # Create scan grid and filter valid data
    cuts = np.arange(range[0], range[1] + incr, incr)
    d_valid = df[df['valid'].values].copy()
    samples = d_valid['sample'].values
    scores = d_valid['BDTscore'].values
    
    # Vectorized efficiency: broadcast cuts against scores
    # Shape: (len(modes), len(cuts))
    print('------>[Info] Plotting Efficiency')
    fig, ax = _make_fig(dpi=dpi)
    try:
        for mode in tqdm(modes):
            mask = samples == mode
            s = scores[mask]
            if len(s) > 0:
                # Vectorized: s[:, None] has shape (n_samples, 1)
                # cuts has shape (len(cuts),)
                # Result: (n_samples, len(cuts)) >= (len(cuts),) 
                eff = np.mean(s[:, None] >= cuts, axis=0)
                ax.plot(cuts, eff, 
                        label=modes_label[mode],
                        color=modes_color[mode])
        
        ax.legend(loc='best', ncols=4)
        ax.set_xlim(*range)
        ax.set_ylim(None, 1.3)
            
        set_labels(ax, 'BDT score', 'Efficiency', 
                right=label, locx=locx, locy=locy)
        savefigs(fig, outDir, 'efficiency', 
                suffix=suffix, format=format)
    finally:
        plt.close(fig)

#_______________________________
def tree_plot(
    bdt: 'xgb.XGBClassifier', 
    inDir: str, 
    outDir: str, 
    epochs: int, 
    n: int, 
    format: list[str] = ['png'], 
    rankdir: str = 'LR'
    ) -> None:
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
    import numpy as np
    import xgboost as xgb

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
        dot = xgb.to_graphviz(
            bdt, num_trees=num_tree, 
            fmap=f'{inDir}/feature.txt', 
            rankdir=rankdir
        )
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

#_______________________________
def hist_check(
    df: 'pd.DataFrame', 
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
    suff: str = 'nominal',
    suffix: str = '',
    format: list[str] = ['png'], 
    htype: str = 'step',
    unity: bool = False, 
    dpi: int = 100,
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
        dpi (int, optional): Figure DPI. Defaults to 100.
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
        strict (bool, optional): If True, clamp x-limits to observed data range. Defaults to True.
    '''
    
    out = f'{outDir}/variables/{suff}'
    mkdir(out)

    # Get splits with single pass
    splits = _get_splits(df)
    sig_modes = [m for m in modes if 'H' in m]
    bkg_modes = [m for m in modes if 'H' not in m]

    # Get data range once
    vdata = df[var].values
    xmin, xmax = vdata.min(), vdata.max()

    fig, ax = _make_fig(dpi=dpi)
    try:
        for linestyle, show_lbl in [('-', True), ('--', False)]:
            sig_data = splits['train_sig'] if linestyle == '-' else splits['val_sig']
            bkg_data = splits['train_bkg'] if linestyle == '-' else splits['val_bkg']
            
            # Plot signal modes
            for mode in sig_modes:
                d = sig_data[sig_data['sample'] == mode]
                if len(d) > 0:
                    wts = _get_wts(d, weight)
                    ax.hist(d[var].values, density=unity, bins=nbins, 
                            range=[xmin, xmax], histtype=htype, 
                            label=modes_label[mode] if show_lbl else None, 
                            color=modes_color[mode],
                            linestyle=linestyle, linewidth=1.5, weights=wts)

            # Stack background samples
            bkg_arrays, bkg_wts, bkg_lbls, bkg_cols = [], [], [], []
            for mode in bkg_modes:
                d = bkg_data[bkg_data['sample'] == mode]
                if len(d) > 0:
                    bkg_arrays.append(d[var].values)
                    bkg_cols.append(modes_color[mode])
                    if show_lbl:
                        bkg_lbls.append(modes_label[mode])
                    if weight:
                        bkg_wts.append(d['weights'].values)

            if bkg_arrays:
                wts_stacked = bkg_wts if weight else None
                lbls_stacked = bkg_lbls if show_lbl else None
                ax.hist(bkg_arrays, density=unity, bins=nbins, 
                        range=[xmin, xmax], histtype=htype, 
                        label=lbls_stacked, color=bkg_cols,
                        stacked=True, linestyle=linestyle, 
                        linewidth=1.5, weights=wts_stacked)
        
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
