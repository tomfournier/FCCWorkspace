from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import ROOT

from constants import h_labels


def Bias(
    df: 'pd.DataFrame',
    nomDir: str,
    outputdir: str,
    h_decays: list[str],
    suffix: str = '',
    ecm: int = 240,
    lumi: float = 10.8,
    outName: str = 'bias',
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


def hist_to_arrays(
    hist: 'ROOT.TH1'
     ) -> tuple['np.ndarray',
                'np.ndarray',
                'np.ndarray']:
    '''Convert a ROOT histogram into numpy arrays of counts, bin edges, and errors.

    Args:
        hist (ROOT.TH1): A ROOT histogram object to convert.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - counts: Array of bin contents (histogram values).
            - bin_edges: Array of bin edges with length = nbins + 1.
            - errors: Array of bin errors (statistical uncertainties).

    Notes:
        The bin_edges array has one more element than counts and errors arrays,
        as it includes both lower edge of the first bin and upper edge of the last bin.
    '''
    import numpy as np

    # Extract bin information from ROOT histogram.
    nbins = hist.GetNbinsX()
    bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, nbins+2)])
    counts    = np.array([hist.GetBinContent(i) for i in range(1, nbins+1)])
    errors    = np.array([hist.GetBinError(i)   for i in range(1, nbins+1)])

    return counts, bin_edges, errors

# ________________________________
def PseudoRatio(
    inDir: str,
    outDir: str,
    cat: str,
    target: str,
    procs: list[str],
    ecm: int = 240,
    lumi: float = 10.8,
    pert: float = 1.05,
    sel: str = '',
    outName: str = 'PseudoRatio',
    format: list[str] = ['png'],
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
    from ..func.bias import hist_from_datacard

    hist_sig, hist_pseudo = hist_from_datacard(inDir, target, cat, procs)

    cut = int(hist_sig.axes[0].edges[-1]/2)

    if 'low' in sel:
        sig, pseudo = hist_sig[:cut], hist_pseudo[:cut]
    elif 'high' in sel:
        sig, pseudo = hist_sig[cut:], hist_pseudo[cut:]
    else:
        sig, pseudo = hist_sig, hist_pseudo

    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})

    np.seterr(divide="ignore", invalid="ignore")
    ratio = np.where(sig!=0, pseudo.values() / sig.values(), np.nan)
    np.seterr(divide="warn", invalid="warn")
    bins = sig.axes[0].edges

    valid: np.ndarray = bins[:-1][np.isfinite(ratio)]
    xMin, xMax = valid.min(), valid.max()

    r = (pseudo.sum().value / sig.sum().value - 1) * 100

    main_plot = MatplotlibPlot(
        outputdir=outDir, output_subdir='PseudoRatio', selection=sel,
        plotname=outName, suffix=f'_{target}', file_formats=format,
        axes=axes[0]
    )
    ratio_plot = MatplotlibPlot(axes=axes[1], style=False)

    # Create step plot for signal using duplicate values
    sig_vals   = sig.values()
    x_step_sig = np.repeat(bins, 2)[1:-1]
    y_step_sig = np.repeat(sig_vals, 2)
    main_plot.draw([([x_step_sig, y_step_sig, 'plot'], {'label': 'Signal', 'color': 'tab:blue', 'linewidth': 3})])

    B = (bins[:-1] + bins[1:]) / 2
    main_plot.draw([([B, pseudo.values(), np.sqrt(pseudo.variances()), 'errorbar'],
                    {'label': 'Pseudo-signal', 'color': 'black', 'linewidth': 2,
                     'linestyle': '', 'marker': 'o', 'markersize': 4, 'capsize': 3})])
    ratio_plot.draw([([bins[:-1], ratio, 'step'], {'where': 'post', 'linewidth': 2, 'color': 'black'})])
    main_plot.draw([([0, 0, 'plot'], {'linestyle': '', 'marker': ',', 'label': f'Signal increased by {r:.2f} %'})])
    ratio_plot.draw([([1.0, 'axhline'], {'color': 'gray', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.6}),
                     ([pert, 'axhline'], {'color': 'black', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.8})])

    # Set axis ranges for both panels.
    main_plot.configure(xlim=(xMin, xMax), xscale='log' if logX else None,
                        yscale='log' if logY else None, legend=True,
                        legend_kwargs={'loc': 'upper right', 'frameon': True, 'fontsize': 20})
    ratio_plot.configure(xlim=(xMin, xMax), ylim=(0.95, pert * 1.05),
                         xscale='log' if logX else None)
    main_plot.axes.tick_params(labelbottom=False)

    # Configure axis labels with appropriate titles.
    lumi_text = rf'$\sqrt{{s}} = {ecm}$ GeV, {lumi} ab$^{{-1}}$'
    xtitle = r'm$_{\mathrm{recoil}}$ High [GeV]' if 'high' in sel \
        else r'm$_{\mathrm{recoil}}$ Low [GeV]'
    ytitle = 'Normalized to Unity' if density else 'Events'
    ratio_plot.set_labels(xtitle, 'Ratio', left=' ')
    main_plot.set_labels('', ytitle, left=' ', right=lumi_text)
    main_plot.save()