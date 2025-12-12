import os

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from tqdm import tqdm
from sklearn.metrics import auc

from .python.plotter import set_plt_style, set_labels, savefigs
from ..tools.utils import mkdir, Significance, Z

# Set the font, size, etc. of the label, tick, legend, etc.
set_plt_style()

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
    '''Generic function to plot training metrics.'''
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
    
    print('----->[Info] Plotting BDT score')
    params = {
        'Signal Training':       {'l':'-',  'c':'r', 'v': False, 's': True},
        'Background Training':   {'l':'-',  'c':'b', 'v': False, 's': False},
        'Signal Validation':     {'l':'--', 'c':'r', 'v': True,  's': True},
        'Background Validation': {'l':'--', 'c':'b', 'v': True,  's': False},
    }

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
    print('------>[Info] Plotting MVA score')

    train_sig = df[(df['valid'] == False) & (df['isSignal'] == 1)]
    valid_sig = df[(df['valid'] == True)  & (df['isSignal'] == 1)]
    train_bkg = df[(df['valid'] == False) & (df['isSignal'] != 1)]
    valid_bkg = df[(df['valid'] == True)  & (df['isSignal'] != 1)]

    sig_modes = [mode for mode in modes if 'H' in mode]
    bkg_modes = [mode for mode in modes if 'H' not in mode]

    datasets = [
        (train_sig, train_bkg, '-', True),
        (valid_sig, valid_bkg, '--', False)
    ]

    ymax = []
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
    print('------>[Info] Plotting feature importance')

    # Get feature importances and sort them by importance
    importance = bdt.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=False)

    # Get the sorted indices of the variables
    sorted_indices = [int(x[0][1:]) for x in sorted_importance]

    # Get the sorted variable names and their corresponding importances
    sorted_vars   = [vars[i] for i in sorted_indices]
    sorted_values = [x[1] for x in sorted_importance]

    # Update variable names with their LaTeX versions
    sorted_vars_latex = [latex_mapping[var] for var in sorted_vars]

    # Create a DataFrame and plot the feature importances
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
    print('------>[Info] Plotting Significance scan')

    df_sig = df[(df['isSignal'] == 1) & (df['valid'] == True)]
    df_bkg = df[(df['isSignal'] == 0) & (df['valid'] == True)]

    df_Z = Significance(df_sig, df_bkg,
                        column=column,
                        weight=weight, 
                        func=Z, nbins=100)
    max_index=df_Z['Z'].idxmax()
    max_Z = df_Z.loc[max_index, 'Z']

    print(f'max-Z: {max_Z:.3f} cut threshold: [{max_index}]')

    np.savetxt(f'{out_txt}/BDT_cut.txt', [float(max_index)])
    print(f'----->[Info] Wrote BDT cut in {out_txt}/BDT_cut.txt')

    fig, ax = plt.subplots()
    try:
        ax.scatter(df_Z.index, df_Z['Z'], marker='.')
        ax.scatter(max_index, max_Z, color='r', marker='*')
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
    
    print('------>[Info] Calculating efficiencies for each mode')
    BDT_cuts = np.arange(range[0], range[1]+incr, incr)
    df_valid = df[df['valid'] == True]

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
    import graphviz

    mkdir(f'{outDir}/feature')
    if n>epochs:
        n = epochs
        print('You have chosen n>epochs. '
              'To avoid redundancy, '
              'n was put equal to epochs')

    print(f'------>[Info] Plotting structure of BDT')
    num_trees = np.linspace(0, epochs-1, n, dtype=int)
    for num_tree in tqdm(num_trees):
        dot = xgb.to_graphviz(bdt, 
                              num_trees=num_tree, 
                              fmap=f'{inDir}/feature.txt', 
                              rankdir=rankdir)
        dot.save(f'{inDir}/bdt.dot')

        with open(f'{inDir}/bdt.dot') as f: 
            dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
        for pl in format:
            graph.render(f'{outDir}/tmp/BDT_{num_tree}', format=pl)
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
    
    out = f'{outDir}/variables'
    mkdir(out)

    train_sig = df[(df['valid'] == False) & (df['isSignal'] == 1)]
    valid_sig = df[(df['valid'] == True)  & (df['isSignal'] == 1)]
    train_bkg = df[(df['valid'] == False) & (df['isSignal'] != 1)]
    valid_bkg = df[(df['valid'] == True)  & (df['isSignal'] != 1)]

    sig_modes = [mode for mode in modes if 'H' in mode]
    bkg_modes = [mode for mode in modes if 'H' not in mode]

    datasets = [
        (train_sig, train_bkg, '-', True),
        (valid_sig, valid_bkg, '--', False)
    ]

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
        ax.legend(loc='upper center', 
                  shadow=False, ncols=ncols)

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
