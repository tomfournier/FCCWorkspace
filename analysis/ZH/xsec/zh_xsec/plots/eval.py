'''BDT evaluation and training diagnostics visualization tools.

Provides comprehensive plotting utilities for XGBoost model assessment, including
training convergence analysis, model performance metrics, feature importance,
and input variable validation.

Functions:
- **Training Diagnostics**: `plot_metrics())` — Plot metrics across boosting iterations.
- **ROC Analysis**: `roc()`, `plot_roc_curve()` — ROC curves and AUC computation for train/validation splits.
- **BDT Scores**: `bdt_score()`, `mva_score()` — Score distributions with signal/background and train/validation breakdown.
- **Feature Analysis**: `importance()` — Feature importance using XGBoost F-score (split count).
- **Optimization**: `significance()` — Scan BDT score thresholds to find optimal cut for maximal significance.
- **Efficiency**: `efficiency()` — Signal/background efficiency vs. BDT cut threshold.
- **Diagnostics**: `hist_check()` — Validate input variables with histograms and train/validation splits.
- **Tree Inspection**: `tree_plot()` — Visualize subset of boosted decision trees using Graphviz.
- **Helpers**: `_get_splits()`, `draw_modes()` — Internal utilities.

Conventions:
- Uses matplotlib for all plot generation with consistent styling via `set_plt_style()`.
- Supports weighted and unweighted histograms; weights applied when data column exists.
- DataFrame cache optimized for repeated calls with same data: `_splits_cache`, `_cache_id`.
- ROC and significance plots mark optimal points with distinctive markers (stars, vertical lines).
- Feature importance sorted in ascending order for readability in horizontal bar charts.
- Efficiency computed as fraction of events passing BDT score threshold (vectorized for speed).
- Tree visualization uses Graphviz with evenly-spaced tree selection for large boosted models.
- Significance threshold scan saves optimal BDT cut to file for downstream use.

Usage:
- Monitor BDT training convergence and diagnose overfitting via training/validation curves.
- Optimize BDT cut threshold by scanning significance (S/√(S+B)) across score range.
- Validate input variables exhibit expected signal/background separations.
- Inspect decision tree structures for interpretability and physics alignment.
- Evaluate model performance on test/validation sets via ROC and efficiency curves.

Lazy Imports:
- numpy, pandas, xgboost imported only when their specific functions are needed.
- sklearn.metrics imported locally in functions requiring ROC calculations.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from tqdm import tqdm

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xgboost as xgb

from plot.pyplot import MatplotlibPlot
from tools.utils import mkdir, Z, Significance
from logger import get_logger

LOGGER = get_logger(__name__)


########################
### CONFIG AND SETUP ###
########################

plt.ioff()  # Disable interactive mode for faster plotting

# Cache for splits to avoid recomputation if same DataFrame used
_splits_cache = {}
_cache_id = None



########################
### HELPER FUNCTIONS ###
########################

def _get_splits(
    df: 'pd.DataFrame'
     ) -> dict[str,
               'pd.DataFrame']:
    '''Extract and cache train/valid and signal/background splits for reuse.

    Uses DataFrame object id as cache key to avoid recomputation if same DataFrame
    used in multiple calls. Cache holds max 3 DataFrames to limit memory usage.

    Args:
        df: Input DataFrame with 'valid' and 'isSignal' columns

    Returns:
        dict: Precomputed boolean-indexed DataFrames: train_sig, train_bkg, val_sig, val_bkg
    '''
    global _cache_id

    # Use DataFrame memory address as cache key
    df_id = id(df)

    # Return cached result if same DataFrame
    if df_id == _cache_id and df_id in _splits_cache:
        return _splits_cache[df_id]

    # Compute boolean masks explicitly to avoid integer bitwise quirks
    is_valid = df['valid'].to_numpy(dtype=bool, copy=False)
    is_sig  = (df['isSignal'] == 1).to_numpy(copy=False)

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

    # Keep cache small (max 5 DataFrames)
    if len(_splits_cache) > 5:
        _splits_cache.clear()

    return result


def draw_modes(
    p: MatplotlibPlot,
    df: 'pd.DataFrame',
    modes: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    column: str,
    nbins: int,
    range: list[float | int],
    htype: str,
    unity: bool,
    weight: bool
     ) -> None:
    '''Draw signal (overlaid) and stacked background histograms for train (solid) and validation (dashed).'''
    splits = _get_splits(df)
    sig_modes = [m for m in modes if 'H' in m]
    bkg_modes = [m for m in modes if 'H' not in m]
    common = {'density': unity, 'bins': nbins, 'range': range,
              'histtype': htype, 'linewidth': 1.5}

    for split, linestyle, show_lbl in [('train', '-', True), ('val', '--', False)]:
        sig_data, bkg_data = splits[f'{split}_sig'], splits[f'{split}_bkg']
        specs = []

        for mode in sig_modes:
            d = sig_data[sig_data['sample'] == mode]
            if len(d) > 0:
                specs.append(([d[column].values, 'hist'],
                              {**common, 'color': colors[mode], 'linestyle': linestyle,
                               'label': labels[mode] if show_lbl else None,
                               'weights': d['weights'].values if weight else None}))

        bkgs = [(m, bkg_data[bkg_data['sample'] == m]) for m in bkg_modes]
        bkgs = [(m, d) for m, d in bkgs if len(d) > 0]
        if bkgs:
            specs.append(([[d[column].values for _, d in bkgs], 'hist'],{
                **common, 'stacked': True, 'linestyle': linestyle, 'color': [colors[m] for m, _ in bkgs],
                'label': [labels[m] for m, _ in bkgs] if show_lbl else None,
                'weights': [d['weights'].values for _, d in bkgs] if weight else None
            }))
        p.draw(specs)



######################
### MAIN FUNCTIONS ###
######################

def plot_metrics(
    outputdir: str,
    results: dict[str, dict[str, list[float]]],
    x_axis: 'np.ndarray',
    metrics: list[str],
    label: str,
    best_iteration: int,
    locx: str = 'right',
    locy: str = 'top',
    suffix: str = '',
    dpi: int = 100,
    file_formats: list[str] = ['png']
     ) -> None:
    '''Plot training and validation metric evolution over boosting iterations.

    Generic metric plotter used by log_loss(), classification_error(), and AUC().
    Plots two curves (training and validation) with marked optimal iteration.

    Args:
        results (dict[str, dict[str, list[float]]]): XGBoost evals_result dictionary: results['dataset_name'][metric_name] = [values...].
        x_axis (np.ndarray): Boosting iteration numbers for x-axis.
        metric_name (str): Metric key to plot (e.g., 'logloss', 'error', 'auc').
        ylabel (str): Y-axis label text.
        label (str): Text placed on figure (e.g., channel name).
        outputdir (str): Directory where plots are written.
        best_iteration (int): Iteration to mark with vertical line.
        locx (str, optional): Legend horizontal position ('left'/'right'). Defaults to 'right'.
        locy (str, optional): Legend vertical position ('top'/'bottom'). Defaults to 'top'.
        suffix (str, optional): Filename suffix. Defaults to ''.
        dpi (int, optional): Figure DPI. Defaults to 100.
        file_formats (list[str], optional): Output formats ['png', 'pdf']. Defaults to ['png'].
    '''
    for metric in metrics:
        LOGGER.debug(f'Plotting {metrics}')
        metric_name = metric.replace(' ', '').lower()
        p = MatplotlibPlot(outputdir, plotname=metric.replace(' ', '_').lower(),
                           suffix=suffix, file_formats=file_formats, dpi=dpi)
        plots = [([x_axis, results['validation_0'][metric_name], 'plot'], {'label': 'Training',   'linewidth': 2}),
                 ([x_axis, results['validation_1'][metric_name], 'plot'], {'label': 'Validation', 'linewidth': 2}),
                 ([best_iteration, 'axvline'], {'color': 'gray', 'label': 'Optimal tree number'})]
        
        try:
            p.draw(plots)
            p.configure(legend=True)
            p.set_labels('Number of Trees', metric, right=label, locx=locx, locy=locy)
            p.save()
        finally:
            p.close()


def plot_roc_curve(
    df: 'pd.DataFrame',
    column: str,
    thresh: float = 0.,
    axe: plt.Axes | None = None,
    color: str | None = None,
    linestyle: str = '-',
    label: str | None = None
) -> None:
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

    if axe   is None: axe = plt.gca()
    if label is None: label = column

    # Get sig/bkg label and MVA score
    is_sig = df['isSignal'].to_numpy(np.int8, copy=False)
    scores = df[column].to_numpy(np.float32, copy=False)

    # Compute the True/False Positive Rate and the Area under Curve
    fpr, tpr = metrics.roc_curve(is_sig, scores)[0:2]
    roc_auc = auc(fpr, tpr)

    # Keep only values above threshold
    mask = tpr >= thresh
    fpr, tpr = fpr[mask], tpr[mask]

    axe.plot(fpr, tpr, label=label+f', AUC = {roc_auc*100:.2f} %',
             color=color, linestyle=linestyle, linewidth=4)


def roc_curve(
    df: 'pd.DataFrame',
    label: str,
    outputdir: str,
    file_formats: list[str] = ['png'],
    eps: float = 0,
    locx: str = 'right',
    locy: str = 'top',
    suffix: str = '',
    dpi: int = 100,
    figsize: float | int = 12
     ) -> None:
    '''Plot ROC curves for training and validation samples.

    Args:
        df (pd.DataFrame): Dataframe with `BDTscore`, `valid`, and `isSignal` columns.
        label (str): Text label placed on the figure.
        outputdir (str): Output directory for plots.
        eps (float, optional): Minimum TPR threshold to display; hides low-signal region. Defaults to 0.
        locx (str, optional): Horizontal placement for label. Defaults to 'right'.
        locy (str, optional): Vertical placement for label. Defaults to 'top'.
        suffix (str, optional): Optional suffix for filenames. Defaults to ''.
        dpi (int, optional): Figure DPI. Defaults to 100.
        file_formats (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    LOGGER.debug('Plotting ROC curve')
    p = MatplotlibPlot(outputdir, plotname='roc', suffix=suffix,
                       file_formats=file_formats, figsize=(figsize, figsize), dpi=dpi)
    plots = [([[eps, 1],      [eps, 1],    'plot'], {'color': 'r', 'lw': 2, 'linestyle': '--', 'label': 'Random Classifier'}),
             ([[eps, eps, 1], [eps, 1, 1], 'plot'], {'color': 'g', 'lw': 2, 'linestyle': '--', 'label': 'Perfect Classifier'})]

    try:
        plot_roc_curve(df[df['valid'] == True],  'BDTscore', axe=p.axes, label='Validation Sample', thresh=eps)
        plot_roc_curve(df[df['valid'] == False], 'BDTscore', axe=p.axes, label='Training Sample',   thresh=eps,
                       linestyle=(0, (1, 1)))
        p.draw(plots)
        p.configure(legend=True)
        p.set_labels('False Positive Rate', 'True Positive Rate', right=label, locx=locx, locy=locy)
        p.save()
    finally:
        p.close()


def bdt_score(
    df: 'pd.DataFrame',
    label: str,
    outputdir: str,
    file_formats: list[str] = ['png'],
    unity: bool = True,
    nbins: int = 100,
    yscale: str = 'log',
    suffix: str = '',
    htype: str = 'step',
    locx: str = 'right',
    locy: str = 'top',
    range: list[float | int] = [0, 1],
    weight: bool = True,
    dpi: int = 100,
     ) -> None:
    '''Plot BDT score distributions for train/validation signal and background.

    Args:
        df (pd.DataFrame): Dataframe with `BDTscore`, `valid`, `isSignal`, and `weights`.
        label (str): Text placed on the figure (e.g., channel or tag).
        outputdir (str): Directory where plots are written.
        file_formats (list[str], optional): List of output formats to render. Defaults to ['png'].
        unity (bool, optional): Normalize each histogram to unit area. Defaults to True.
        nbins (int, optional): Number of histogram bins. Defaults to 100.
        yscale (str, optional): Y-axis scale (`linear` or `log`). Defaults to 'log'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        htype (str, optional): Matplotlib histogram type. Defaults to 'step'.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        range (list[float | int], optional): Score range used for binning. Defaults to [0, 1].
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
        dpi (int, optional): Figure DPI. Defaults to 100.
    '''
    LOGGER.debug('Plotting BDT score')

    # Get splits with single pass through data
    splits = _get_splits(df)
    df_len = len(df)

    # Define styling: linestyle, color, valid flag, sig flag
    params = [('Signal Training',   '-',  'r', splits['train_sig']), ('Background Training',   '-',  'b', splits['train_bkg']),
              ('Signal Validation', '--', 'r', splits['val_sig']),   ('Background Validation', '--', 'b', splits['val_bkg'])]

    p = MatplotlibPlot(outputdir, plotname='bdt_score', suffix=suffix, file_formats=file_formats, dpi=dpi)
    try:
        specs = []
        for name, ls, c, d in params:
            ratio = (len(d) / df_len) * 100.0
            LOGGER.debug(f'{name:<25} {len(d):6d} Ratio: {ratio:5.2f}%')
            specs.append(([d['BDTscore'].values, 'hist'], {
                'density': unity, 'bins': nbins, 'range': range, 'histtype': htype, 'label': name, 'linestyle': ls,
                'color': c, 'linewidth': 1.5, 'weights': d['weights'].values if weight else None
            }))
        p.draw(specs)

        p.configure(xlim=range, yscale=yscale, legend=True, legend_kwargs={'loc': 'upper right', 'shadow': False})
        yMin, yMax = p.axes.get_ylim()
        p.configure(ylim=(yMin, yMax * (10 if yscale=='log' else 1.25)))

        ylabel = 'Normalized to Unity' if unity else 'Events'
        p.set_labels('BDT Score', ylabel, right=label, locx=locx, locy=locy)
        p.save()
    finally:
        p.close()


def mva_score(
    df: 'pd.DataFrame',
    label: str,
    outputdir: str,
    modes: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    file_formats: list[str] = ['png'],
    unity: bool = True,
    nbins: int = 100,
    locx: str = 'right',
    locy: str = 'top',
    suffix:str = '',
    htype: str = 'step',
    yscale: str = 'log',
    range: list[float | int] = [0, 1],
    ncols: int = 3,
    dpi: int = 100,
    weight: bool = True
     ) -> None:
    '''Plot stacked BDT score histograms for multiple modes.

    Args:
        df (pd.DataFrame): Dataframe with sample labels and BDT scores.
        label (str): Text placed on the figure (e.g., channel or tag).
        outputdir (str): Directory where plots are written.
        modes (list[str]): Ordered list of sample names.
        colors (dict[str, str]): Mapping from mode to color.
        labels (dict[str, str]): Mapping from mode to legend label.
        file_formats (list[str], optional): List of output formatss to render. Defaults to ['png'].
        unity (bool, optional): Normalize each histogram to unit area. Defaults to True.
        nbins (int, optional): Number of histogram bins. Defaults to 100.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        htype (str, optional): Matplotlib histogram type. Defaults to 'step'.
        yscale (str, optional): Y-axis scale (`linear` or `log`). Defaults to 'log'.
        range (list[float | int], optional): Score range used for binning. Defaults to [0, 1].
        ncols (int, optional): Number of legend columns. Defaults to 3.
        dpi (int, optional): Figure DPI. Defaults to 100.
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
    '''
    LOGGER.debug('Plotting MVA score')

    p = MatplotlibPlot(outputdir, plotname='mva_score', suffix=suffix,
                       file_formats=file_formats, dpi=dpi)
    try:
        draw_modes(p, df, modes, colors, labels, 'BDTscore', nbins, range, htype, unity, weight)

        p.configure(range, yscale=yscale)
        yMin, yMax = p.axes.get_ylim()
        p.configure(ylim=(yMin, yMax * (10 if yscale=='log' else 1.25)), legend=True,
                    legend_kwargs={'loc': 'upper right', 'shadow': False, 'ncols': ncols})

        ylabel = 'Normalized to Unity' if unity else 'Events'
        p.set_labels('BDT Score', ylabel, right=label, locx=locx, locy=locy)
        p.save()
    finally:
        p.close()


def importance(
    bdt: 'xgb.XGBClassifier',
    label: str,
    outputdir: str,
    input_vars: list[str],
    vars_label: dict[str, str],
    file_formats: list[str] = ['png'],
    locx: str = 'right',
    locy: str = 'top',
    suffix: str = '',
    dpi: int = 100,
     ) -> None:
    '''Plot feature importance using the XGBoost F-score (split count).

    Args:
        bdt (xgb.XGBClassifier): Trained XGBClassifier instance.
        vars (list[str]): Feature names ordered as used in training.
        latex_mapping (dict[str, str]): Mapping from feature name to display label.
        label (str): Text placed on the figure (e.g., channel or tag).
        outputdir (str): Directory where plots are written.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        dpi (int, optional): Figure DPI. Defaults to 100.
        file_formats (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    LOGGER.debug('Plotting feature importance')

    import pandas as pd

    # Extract and sort feature importances by F-score (split count)
    importance: dict = bdt.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1])

    # Map feature indices to LaTeX labels and extract values in single pass
    sorted_vars   = [vars_label[input_vars[int(k[1:])]] for k, _ in sorted_importance]
    sorted_values = [v for _, v in sorted_importance]

    # Build DataFrame and create horizontal bar chart
    importance_df = pd.DataFrame({'Variable': sorted_vars, 'Importance': sorted_values})

    p = MatplotlibPlot(outputdir, plotname='importance', suffix=suffix, file_formats=file_formats, dpi=dpi)
    try:
        importance_df.plot(kind='barh', x='Variable', y='Importance', legend=None, ax=p.axes)
        p.set_labels('F-score', 'Variables', right=label, locx=locx, locy=locy)
        p.axes.grid(False, axis='y')
        p.save()
    finally:
        p.close()


def significance(
    df: 'pd.DataFrame',
    label: str,
    outputdir: str,
    out_txt: str,
    file_formats: list[str] = ['png'],
    locx: str = 'right',
    locy: str = 'top',
    column: str = 'BDTscore',
    weight: str = 'norm_weight',
    suffix: str = '',
    dpi: int = 100,
     ) -> None:
    '''Scan the BDT score for maximal significance and save the cut value.

    Args:
        df (pd.DataFrame): Dataframe with scores, weights, and labels.
        label (str): Text placed on the figure (e.g., channel or tag).
        outputdir (str): Directory where plots are written.
        out_txt (str): Directory where the optimal cut is stored.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        column (str, optional): Score column to scan. Defaults to 'BDTscore'.
        weight (str, optional): Weight column used in the significance calculation. Defaults to 'norm_weight'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        dpi (int, optional): Figure DPI. Defaults to 100.
        file_formats (list[str], optional): List of output formats to render. Defaults to ['png'].
    '''
    LOGGER.debug('Plotting Significance scan')

    # Extract validation signal and background samples
    df_sig = df[(df['isSignal'] == 1) & (df['valid'] == True)]
    df_bkg = df[(df['isSignal'] == 0) & (df['valid'] == True)]

    # Compute significance across BDT score thresholds
    df_Z = Significance(df_sig, df_bkg, column, weight, Z, nbins=100)
    max_idx = df_Z['Z'].idxmax()
    max_Z = df_Z.loc[max_idx, 'Z']

    LOGGER.debug(f'Max Z: {max_Z:,.3f}, cut threshold: {max_idx:,.2f}')

    # Save optimal BDT cut threshold to file
    with open(f'{out_txt}/BDT_cut{suffix}.txt', 'w') as out:
        out.write(f'{max_idx:.2f}')
    LOGGER.debug(f'Wrote BDT cut at {out_txt}/BDT_cut.txt')

    p = MatplotlibPlot(outputdir, plotname='significance_scan', suffix=suffix, file_formats=file_formats, dpi=dpi)
    plots = [([df_Z.index, df_Z['Z'], 'scatter'], {'marker': '.'}),
             ([[max_idx], [max_Z],  'scatter'], {'color': 'r', 'marker': '*', 's': 150}),
             ([max_idx, 'axvline'], {'color': 'gray', 'alpha': 0.8, 'label': f'max-Z: {max_Z:.3f}\ncut threshold: [{max_idx:.2f}]'})]
    try:
        p.draw(plots)
        p.configure(legend=True)
        p.set_labels('BDT score', 'Significance', right=label, locx=locx, locy=locy)
        p.save()
    finally:
        p.close()


def efficiency(
    df: 'pd.DataFrame',
    label: str,
    outputdir: str,
    modes: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    incr: float = 0.01,
    file_formats: list[str] = ['png'],
    locx: str = 'right',
    locy: str = 'top',
    suffix: str = '',
    range: list[int | float] = [0, 1],
    dpi: int = 100,
     ) -> None:
    '''Compute and plot efficiency vs. BDT cut for each mode.

    Args:
        df (pd.DataFrame): Dataframe with scores and sample labels.
        modes (list[str]): List of sample names to process.
        modes_label (dict[str, str]): Mapping from mode to legend label.
        modes_color (dict[str, str]): Mapping from mode to color.
        label (str): Text placed on the figure (e.g., channel or tag).
        outputdir (str): Directory where plots are written.
        locx (str, optional): Horizontal placement for the right-side label. Defaults to 'right'.
        locy (str, optional): Vertical placement for the right-side label. Defaults to 'top'.
        suffix (str, optional): Optional suffix appended to filenames. Defaults to ''.
        file_formats (list[str], optional): List of output formats to render. Defaults to ['png'].
        range (list[int | float], optional): Inclusive BDT score range to scan. Defaults to [0, 1].
        dpi (int, optional): Figure DPI. Defaults to 100.
        incr (float, optional): Step size for BDT cut thresholds. Defaults to 0.01.
    '''
    import numpy as np
    LOGGER.debug('Calculating efficiencies for each mode')

    # Create scan grid and filter valid data
    cuts = np.arange(range[0], range[1] + incr, incr)
    d_valid: pd.DataFrame = df[df['valid'].values].copy()
    samples = d_valid['sample'].values
    scores  = d_valid['BDTscore'].values

    # Vectorized efficiency: broadcast cuts against scores
    # Shape: (len(modes), len(cuts))
    LOGGER.debug('Plotting Efficiency')
    p = MatplotlibPlot(outputdir, plotname='efficiency', suffix=suffix, file_formats=file_formats, dpi=dpi)
    try:
        specs = []
        for mode in tqdm(modes):
            s = scores[samples == mode]
            if len(s) > 0:
                # Fraction of events passing each cut, shape (len(cuts),)
                eff = np.mean(s[:, None] >= cuts, axis=0)
                specs.append(([cuts, eff, 'plot'], {'label': labels[mode], 'color': colors[mode]}))
        p.draw(specs)

        p.configure(xlim=range, ylim=(None, 1.4), legend=True, legend_kwargs={'loc': 'best', 'ncols': 3})
        p.set_labels('BDT score', 'Efficiency', right=label, locx=locx, locy=locy)
        p.save()
    finally:
        p.close()


def tree_plot(
    bdt: 'xgb.XGBClassifier',
    inputdir: str,
    outputdir: str,
    epochs: int,
    n: int,
    file_formats: list[str] = ['png'],
    rankdir: str = 'LR'
     ) -> None:
    '''Render a subset of boosted trees to Graphviz files.

    Args:
        bdt (xgb.XGBClassifier): Trained XGBClassifier instance.
        inDir (str): Directory containing the feature map file.
        outputdir (str): Directory where rendered trees are stored.
        epochs (int): Total number of boosting iterations.
        n (int): Number of trees to render (evenly spaced up to epochs).
        file_formats (list[str], optional): List of output formats to render. Defaults to ['png'].
        rankdir (str, optional): Graphviz rank direction. Defaults to 'LR'.
    '''
    import graphviz, subprocess
    import numpy as np
    import xgboost as xgb

    mkdir(f'{outputdir}/feature')
    # Ensure n does not exceed total number of epochs
    if n > epochs:
        n = epochs
        LOGGER.warning('n > epochs: resetting n to match total epoch')

    LOGGER.debug('Plotting structure of BDT')
    # Generate evenly-spaced tree indices for visualization
    num_trees = np.linspace(0, epochs-1, n, dtype=int)
    for num_tree in tqdm(num_trees):
        # Export tree to DOT format
        dot = xgb.to_graphviz(bdt, f'{inputdir}/feature.txt', num_tree, rankdir)
        dot.save(f'{inputdir}/bdt.dot')

        # Render to requested formats using graphviz
        with open(f'{inputdir}/bdt.dot') as f:
            dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
        for pl in file_formats:
            graph.render(f'{outputdir}/tmp/BDT_{num_tree}', format=pl)
    
    # Move rendered images to output directory and clean up
    subprocess.run(['mv', f'{outputdir}/tmp/*.png', f'{outputdir}/feature/'])
    subprocess.run(['rm', '-rf', f'{outputdir}/tmp'])
    LOGGER.info(f'Plotted structure of BDT in {outputdir}/feature')


def hist_check(
    df: 'pd.DataFrame',
    label: str,
    outputdir: str,
    modes: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    column: str,
    xlabel: str,
    file_formats: list[str] = ['png'],
    nbins: int = 100,
    ncols: int = 3,
    yscale: str = 'linear',
    locx: str = 'right',
    locy: str = 'top',
    suff: str = 'nominal',
    suffix: str = '',
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
        outputdir (str): Directory where plots are written.
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
        file_formats (list[str], optional): List of output formats to render. Defaults to ['png'].
        htype (str, optional): Matplotlib histogram type. Defaults to 'step'.
        unity (bool, optional): Normalize each histogram to unit area. Defaults to False.
        dpi (int, optional): Figure DPI. Defaults to 100.
        weight (bool, optional): Whether to apply per-event weights. Defaults to True.
        strict (bool, optional): If True, clamp x-limits to observed data range. Defaults to True.
    '''

    # Get data range once
    vdata = df[column].values
    xmin, xmax = vdata.min(), vdata.max()

    p = MatplotlibPlot(outputdir, f'variables/{suff}', plotname=column,
                       suffix=suffix, file_formats=file_formats, dpi=dpi)
    try:
        draw_modes(p, df, modes, colors, labels, column, nbins, [xmin, xmax], htype, unity, weight)

        # Constrain x-limits to data range if strict mode enabled
        p.configure((xmin, xmax) if strict else None, yscale=yscale)
        yMin, yMax = p.axes.get_ylim()
        p.configure(ylim=(yMin, yMax * (10 if yscale=='log' else 1.25)), legend=True,
                    legend_kwargs={'loc': 'upper center', 'shadow': False, 'ncols': ncols})

        ylabel = 'Normalized to Unity' if unity else 'Events'
        p.set_labels(xlabel, ylabel, right=label, locx=locx, locy=locy)
        p.save()
    finally:
        p.close()
