from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from constants import h_labels


def Bias(
    df: 'pd.DataFrame',
    nomDir: str,
    outputdir: str,
    h_decays: list[str],
    ecm: int = 240,
    lumi: float = 10.8,
    outName: str = 'bias',
    suffix: str = '',
    file_formats: list[str] = ['png'],
    quiet: bool = False
     ) -> None:
    '''Plot bias distribution per Higgs decay with uncertainty bands.

    Creates a scatter plot showing bias values for each Higgs decay mode,
    with markers inside/outside the uncertainty band to indicate compatibility.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'mode' and 'bias' with decay mode names and bias values respectively.
        nomDir (str): Path to nominal results directory containing 'results.txt'.
        outDir (str): Path for output plots.
        h_decays (list[str]): Higgs decay mode names (e.g., ['bb', 'WW', 'ZZ', 'tautau']).
        suffix (str, optional): Filename suffix. Defaults to ''.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        outName (str, optional): Base output filename. Defaults to 'bias'.
        format (list[str], optional): Image formats. Defaults to ['png'].
    '''

    import json, ROOT
    from plot.pullplot import PullPlot

    # Load bias values and uncertainty from results file.
    bias_dict = dict(zip(df['mode'], df['bias'] * 100))
    with open(f'{nomDir}/results_fit.json', 'r') as fIn:
        unc = json.load(fIn)['r']['err'] * 1e4

    bias_values = list(bias_dict.values())
    max_bias = max(abs(min(bias_values)), abs(max(bias_values)))
    # Set axis limits to accommodate uncertainty bands with 20% margin.
    xMin, xMax = (-int(unc*1.2), int(unc*1.2)) if unc > max_bias \
        else (-int(max_bias*1.2), int(max_bias*1.2))

    labels = [h_labels[h_decay] for h_decay in h_decays]
    colors = [ROOT.kBlack if abs(bias_dict[h_decay]) < unc else ROOT.kRed for h_decay in h_decays]
    plot = PullPlot(outputdir, None, None,
                    [bias_dict[h_decay] for h_decay in h_decays],
                    [0.] * len(h_decays), labels, 0, unc,
                    ecm, lumi, 'Bias (#times 100) [%]', 'pulls',
                    False, False)
    plot.draw(outName, suffix, file_formats, quiet, xMin, xMax, point_colors=colors)


def BiasNorm(
    df: 'pd.DataFrame',
    nomDir: str,
    outputdir: str,
    h_decays: list[str],
    ecm: int = 240,
    lumi: float = 10.8,
    outName: str = 'bias_norm',
    suffix: str = '',
    file_formats: list[str] = ['png'],
    quiet: bool = False
     ) -> None:
    '''Same as `Bias`, with the biases normalised to the fit uncertainty (band at +-1).'''

    import json, ROOT
    from plot.pullplot import PullPlot

    with open(f'{nomDir}/results_fit.json', 'r') as fIn:
        unc = json.load(fIn)['r']['err'] * 1e4

    bias_dict = dict(zip(df['mode'], df['bias'] * 100 / unc))

    # Axis limits with 20% margin around the band and the largest bias
    lim = 1.2 * max(1, max(abs(b) for b in bias_dict.values()))

    labels = [h_labels[h_decay] for h_decay in h_decays]
    colors = [ROOT.kBlack if abs(bias_dict[h_decay]) < 1 else ROOT.kRed for h_decay in h_decays]
    plot = PullPlot(outputdir, None, None,
                    [bias_dict[h_decay] for h_decay in h_decays],
                    [0.] * len(h_decays), labels, 0, 1,
                    ecm, lumi, 'Bias /#Delta#sigma_{ZH}', 'pulls',
                    False, False)
    plot.draw(outName, suffix, file_formats, quiet, -lim, lim, point_colors=colors)


def BiasSorted(
    df: 'pd.DataFrame',
    nomDir: str,
    outputdir: str,
    h_decays: list[str],
    ecm: int = 240,
    lumi: float = 10.8,
    outName: str = 'bias_sorted',
    suffix: str = '',
    file_formats: list[str] = ['png'],
    quiet: bool = False
     ) -> None:
    '''Same as `BiasNorm`, with the decays sorted by |bias|: the lowest at the bottom, the highest at the top.'''

    import json, ROOT
    from plot.pullplot import PullPlot

    with open(f'{nomDir}/results_fit.json', 'r') as fIn:
        unc = json.load(fIn)['r']['err'] * 1e4
    
    bias_dict = dict(zip(df['mode'], df['bias'] * 100 / unc))

    # Set axis limits to accommodate uncertainty bands with 20% margin.
    lim = 1.2 * max(1, max(abs(b) for b in bias_dict.values()))

    # The first entry of PullPlot is drawn at the bottom
    h_sorted = sorted(h_decays, key=lambda h: abs(bias_dict[h]))
    labels = [h_labels[h] for h in h_sorted]
    colors = [ROOT.kBlack if abs(bias_dict[h]) < 1 else ROOT.kRed for h in h_sorted]
    plot = PullPlot(outputdir, None, None,
                    [bias_dict[h] for h in h_sorted],
                    [0.] * len(h_sorted), labels, 0, 1,
                    ecm, lumi, 'Bias /#Delta#sigma_{ZH}', 'pulls',
                    False, False)
    plot.draw(outName, suffix, file_formats, quiet, -lim, lim, point_colors=colors)


def hist_from_datacard(
    inDir: str,
    target: str,
    cat: str,
    procs: list[str]
     ) -> tuple['np.ndarray',
                'np.ndarray',
                'np.ndarray',
                'np.ndarray']:
    '''Read the signal and the background-subtracted pseudo-data from a datacard ROOT file.

    Args:
        inDir: Directory containing the datacard ROOT file.
        target: Target channel name matching the datacard file suffix.
        cat: Category name as in the datacard (e.g., 'ee_high').
        procs: Process names (signal first, then backgrounds).

    Returns:
        (bin_edges, signal, pseudo_minus_bkg, pseudo_variance)
    '''
    import uproot

    with uproot.open(f'{inDir}/datacard_{target}.root') as f:
        pseudo = f[f'z_{cat}_asimov']
        bkgs   = [f[f'z_{cat}_{proc}'] for proc in procs[1:]]
        edges  = pseudo.axis().edges()
        sig    = f[f'z_{cat}_{procs[0]}'].values()
        values = pseudo.values()    - sum(h.values() for h in bkgs)
        var    = pseudo.variances() + sum(h.variances() for h in bkgs)

    return edges, sig, values, var


def PseudoRatio(
    inputdir: str,
    outputdir: str,
    target: str,
    procs: list[str],
    cat: str,
    ecm: int = 240,
    selection: str = '',
    lumi: float = 10.8,
    pert: float = 1.05,
    outName: str = 'PseudoRatio',
    file_formats: list[str] = ['png'],
    logX: bool = False,
    logY: bool = False,
    density: bool = False,
     ) -> None:
    '''Compare nominal signal to pseudo-signal and plot ratio with errors.

    This function creates a publication-quality plot comparing the nominal signal
    distribution to a pseudo-signal (background-subtracted data or alternative
    signal scenario) and displays their ratio in a two-panel figure.

    Args:
        variable (str): Name of the variable to plot (e.g., histogram key in input files).
        inDir (str): Path to input directory containing histogram files.
        outDir (str): Path to output directory where plots will be saved.
        cat (str): Category identifier for pseudo-signal generation.
        target (str): Target process or decay channel identifier.
        z_decays (list[str]): List of Z boson decay modes (e.g., ['ee', 'mumu', 'tautau']).
        h_decays (list[str]): List of Higgs boson decay modes (e.g., ['bb', 'WW', 'ZZ']).
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        pert (float, optional): Perturbation factor for pseudo-signal variation. Defaults to 1.05.
        sel (str, optional): Selection criteria tag (e.g., 'high', 'low'). Defaults to ''.
        rebin (int, optional): Rebinning factor for histograms. Defaults to 1.
        outName (str, optional): Base name for output files. Defaults to 'PseudoRatio'.
        format (list[str], optional): Image format(s) for saved plots. Defaults to ['png'].
        proc_scales (dict[str, float], optional): Process-specific scaling factors. Defaults to {}.
        logX (bool, optional): Apply logarithmic scaling to x-axis. Defaults to False.
        logY (bool, optional): Apply logarithmic scaling to y-axis. Defaults to False.
        lazy (bool, optional): Use lazy loading for histograms. Defaults to True.
        tot (bool, optional): Include total contributions in pseudo-signal. Defaults to True.
        density (bool, optional): Normalize both histograms to unit area. Defaults to False.
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from plot.pyplot import MatplotlibPlot

    bins, sig, pseudo, var = hist_from_datacard(inputdir, target, cat, procs)

    ratio = np.divide(pseudo, sig, out=np.full_like(pseudo, np.nan), where=sig != 0)
    xMin, xMax = bins[:-1][np.isfinite(ratio)][[0, -1]]
    r = (pseudo.sum() / sig.sum() - 1) * 100

    _, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    main_plot = MatplotlibPlot(outputdir, 'PseudoRatio', selection, outName,
                               f'_{target}', file_formats, axes=axes[0])
    ratio_plot = MatplotlibPlot(axes=axes[1], style=False)

    # Create step plot for signal using duplicate values
    x_step_sig = np.repeat(bins, 2)[1:-1]
    y_step_sig = np.repeat(sig, 2)
    main_plot.draw([([x_step_sig, y_step_sig, 'plot'], {'label': 'Signal', 'color': 'tab:blue', 'linewidth': 3})])

    B = (bins[:-1] + bins[1:]) / 2
    main_plots = [([B, pseudo, np.sqrt(var), 'errorbar'],
                   {'label': 'Pseudo-signal', 'color': 'black', 'linewidth': 2,
                    'linestyle': '', 'marker': 'o', 'markersize': 4, 'capsize': 3}),
                   ([0, 0, 'plot'], {'linestyle': '', 'marker': ',', 'label': f'Signal increased by {r:.2f} %'})]
    ratio_plots = [([bins[:-1], ratio, 'step'], {'where': 'post', 'linewidth': 2, 'color': 'black'}),
                   ([1.0,  'axhline'], {'color': 'gray',  'linestyle': '-',  'linewidth': 2, 'alpha': 0.6}),
                   ([pert, 'axhline'], {'color': 'black', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8})]
    main_plot.draw(main_plots)
    ratio_plot.draw(ratio_plots)

    # Set axis ranges for both panels.
    main_plot.configure(xlim=(xMin, xMax), xscale='log' if logX else None, yscale='log' if logY else None,
                        legend=True, legend_kwargs={'loc': 'upper right', 'frameon': True, 'fontsize': 20})
    ratio_plot.configure(xlim=(xMin, xMax), ylim=(0.95, pert * 1.05), xscale='log' if logX else None)
    main_plot.axes.tick_params(labelbottom=False)

    # Configure axis labels with appropriate titles.
    lumi_text = rf'$\sqrt{{s}} = {ecm}$ GeV, {lumi} ab$^{{-1}}$'
    xtitle = r'm$_{\mathrm{recoil}}$ SEL [GeV]'.replace('SEL', 'High' if 'high' in selection else 'Low')
    ytitle = 'Normalized to Unity' if density else 'Events'
    ratio_plot.set_labels(xtitle, 'Ratio', left=' ')
    main_plot.set_labels('', ytitle, left=' ', right=lumi_text)
    main_plot.save()