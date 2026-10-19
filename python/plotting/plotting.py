'''Publication-quality histogram and distribution plotting utilities.

Provides:
- Signal/background visualization: `makePlot()`, `PlotDecays()`, `makePlot()`.
- Cut optimization plots: `significance()`.
- Yields summary canvases: `AAAyields()`.
- Bias and pseudo-data analysis: `Bias()`, `PseudoRatio()`.
- Flexible argument processing: `get_args()`, `args_decay()`, `_extract_nested_args()`.
- Directory and utility helpers: `_parse_selection_dir()`, `_ensure_plt_style()`.

Functions:
- `makePlot()`: Draw signal/background histograms with optional stacking and scaling.
- `PlotDecays()`: Compare Higgs decay modes with unit-integral normalization.
- `significance()`: Plot running significance and signal efficiency vs. cut value.
- `AAAyields()`: Render yields summary canvas with process yields and metadata.
- `Bias()`: Plot bias distributions per Higgs decay mode with uncertainty bands.
- `PseudoRatio()`: Create ratio plots comparing nominal and pseudo-signal distributions.
- `get_args()`: Extract plotting arguments with hierarchical lookup and wildcard matching.
- `args_decay()`: Extract decay-specific plotting arguments (excludes 'make' mode).

Argument Hierarchies:
- Supports three-level nesting: args[var] → args[var][ecm] → args[var][ecm][sel]
- Wildcard pattern matching: *pattern*, *pattern, pattern* for sel key matching.
- Pipe-separated selection lists: 'sel1|sel2' for matching multiple selections.
- Auto-populated defaults for xmin, xmax, ymin, ymax, rebin, lumi, ecm, etc.

Conventions:
- Uses ROOT graphics backend for histograms; matplotlib for significance plots.
- Consistent styling via imported helper modules (plotter, helper).
- Output organized by selection (nominal/high/low) and subplot directories.
- All plots support both linear and logarithmic axes.

Usage:
- Create publication-ready plots with automatic styling and legend management.
- Scan cut values for optimal significance in physics analyses.
- Compare process or decay mode distributions with flexible scaling and normalization.

Lazy Imports:
- ROOT, numpy, pandas loaded via local imports only when needed (function-level).
- Configuration and styling from package modules loaded once at import.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from inspect import Parameter, signature
from typing import Any, Union

from constants import h_labels
from logger import get_logger

LOGGER = get_logger(__name__)



########################
### CONFIG AND SETUP ###
########################

# Tracks whether the matplotlib style has been set for this session.
PLT_STYLE_SET = False



########################
### HELPER FUNCTIONS ###
########################

def _extract_nested_args(
    var_args: dict[str, dict | float | int | str],
    ecm: int,
    sel: str,
    function
     ) -> dict[str, float | int | str]:
    '''Navigate nested args structure to extract parameters for given ecm and sel.

    Supports flexible hierarchical nesting with wildcard selection matching:
    1. args[var] = {...parameters...}  # Direct flat parameters
    2. args[var][ecm] = {...parameters...}  # Organized by ecm (int key)
    3. args[var][ecm][sel_pattern] = {...parameters...}  # By ecm and sel pattern
    4. args[var][sel_pattern] = {...parameters...}  # By sel pattern only

    Selection Pattern Matching (applied in order):
    - Exact match: sel == pattern
    - Pipe-separated lists: 'sel1|sel2|*pattern*' with wildcard support
    - Wildcard patterns: '*pattern*' (contains), '*pattern' (ends), 'pattern*' (starts)

    Args:
        var_args (dict): The args dictionary for a specific variable.
        ecm (int): Center-of-mass energy in GeV.
        sel (str): Selection criteria identifier to match.
        function: Plotting function whose parameters define valid option keys.

    Returns:
        dict: Copy of matched parameters dict, or empty dict if no match found.
    '''

    current = var_args

    # Try to navigate by ecm (as integer key) if present
    if ecm in current: current = current[ecm]

    # Try to find matching sel pattern in current level
    # Function parameters identify option keys; all other string keys are patterns.
    param_keys = set(signature(function).parameters)

    for key in current.keys():
        if isinstance(key, str) and key not in param_keys:
            # This looks like a selection pattern, try to match it
            if key == sel:
                # Exact match
                return current[key].copy()
            elif '|' in key:
                # Pipe-separated list of selections (e.g., 'Baseline_sep|Baseline_high')
                patterns = [p.strip() for p in key.split('|')]
                for pattern in patterns:
                    if pattern == sel:
                        # Exact match within pipe-separated list
                        return current[key].copy()
                    elif '*' in pattern:
                        # Wildcard within pipe-separated list
                        if pattern.startswith('*') and pattern.endswith('*'):
                            if pattern[1:-1] in sel:
                                return current[key].copy()
                        elif pattern.startswith('*'):
                            if sel.endswith(pattern[1:]):
                                return current[key].copy()
                        elif pattern.endswith('*'):
                            if sel.startswith(pattern[:-1]):
                                return current[key].copy()
            elif '*' in key:
                # Wildcard pattern matching
                if key.startswith('*') and key.endswith('*'):
                    # *pattern* - contains
                    if key[1:-1] in sel:
                        return current[key].copy()
                elif key.startswith('*'):
                    # *pattern - ends with
                    if sel.endswith(key[1:]):
                        return current[key].copy()
                elif key.endswith('*'):
                    # pattern* - starts with
                    if sel.startswith(key[:-1]):
                        return current[key].copy()

    # No sel-specific match found
    # Return current level if it contains parameter keys (not just nested dicts)
    if any(k in param_keys for k in current.keys()):
        return current.copy()

    # Return empty dict if no parameters found
    return {}

def decay_plots(process: str, ecm: int) -> dict[str, dict[str, tuple[str, ...]]]:
    from samples import get_process_dict
    from constants import H_decays
    return {'signals': {d: get_process_dict([process], ecm, h_decays=[d])[process] for d in H_decays}}


def plot_configs(ecm: int, cat: str) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    from samples import get_process_dict
    backgrounds = get_process_dict(['WW', 'ZZ', 'Zgamma', 'Rare'] + (['tt'] if cat == 'qq' and ecm == 365 else []), ecm)
    category = {'signals': get_process_dict([f'Z{cat}H'],       ecm), 'backgrounds': backgrounds}
    procs    = {'signals': get_process_dict(['ZH', f'Z{cat}H'], ecm), 'backgrounds': backgrounds}
    total    = {'signals': get_process_dict(['ZH'],             ecm), 'backgrounds': backgrounds}
    return {'category': category, 'total': total, 'procs': procs, 'decay': decay_plots(f'Z{cat}H', ecm), 'total_decay': decay_plots('ZH', ecm)}

def plot_configs_mass(ecm: int, cat: str) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    from samples import get_process_dict_mass
    backgrounds = get_process_dict_mass(['WW', 'ZZ', 'Zgamma', 'Rare'], ecm)
    category = {'signals': get_process_dict_mass([f'Z{cat}H'],       ecm), 'backgrounds': backgrounds}
    procs    = {'signals': get_process_dict_mass(['ZH', f'Z{cat}H'], ecm), 'backgrounds': backgrounds}
    total    = {'signals': get_process_dict_mass(['ZH'],             ecm), 'backgrounds': backgrounds}
    return {'category': category, 'total': total, 'procs': procs}


def get_args(
    var: str,
    function,
    args: dict[str, dict[str, str | float | int]],
    context: dict[str, Any] | None = None,
    **overrides: Any
     ) -> dict[str, Any]:
    '''Extract and merge plotting arguments for variable/selection with defaults.

    Retrieves user-provided plotting options, applies selection and energy filters,
    resolves the `which` flag for plot type, and populates all missing parameters
    with sensible defaults.

    Hierarchical Lookup:
    - args[var] = {...params...}  # Direct parameters
    - args[var][ecm] = {...params...}  # By center-of-mass energy
    - args[var][ecm][sel_pattern] = {...params...}  # By energy and selection
    - args[var][sel_pattern] = {...params...}  # By selection only

    Selection Patterns:
    - Exact match: sel == pattern
    - Wildcard: 'Baseline*' matches 'Baseline', 'Baseline_high', 'Baseline_low'
    - Pipe-separated: 'Baseline_sep|Baseline_high' matches either pattern

    Filter Keys:
    - 'which' (str): 'both', 'make', 'decay' — filtered before defaults applied
    - 'sel' (str): Selection filter; '*' wildcard supported
    - 'ecm' (int): Energy filter; non-matching args cleared

    Args:
        var (str): Variable name to retrieve arguments for.
        sel (str): Selection criteria identifier.
        ecm (int): Center-of-mass energy in GeV.
        lumi (float): Integrated luminosity in ab^-1.
        args (dict[str, dict[str, Union[str, float, int]]]): Nested dictionary of plotting arguments.

    Returns:
        dict[str, Union[str, float, int]]: Complete plotting config with all keys populated.
    '''

    context = {**(context or {}), **overrides}
    sel, ecm, lumi = context['selection'], context['ecm'], context['lumi']
    parameters = signature(function).parameters
    raw = _extract_nested_args(args[var], ecm, sel, function) if var in args else {}
    defaults = {name: parameter.default
                for name, parameter in parameters.items()
                if parameter.default is not Parameter.empty}

    which = raw.get('which', 'both')
    mode = 'decay' if function is PlotDecays else 'make'
    if which not in ('both', mode): raw = {}
    if which not in ('both', 'make', 'decay'):
        LOGGER.warning("Wrong value given to 'which', acting as 'both'")

    if 'sel' in raw:
        match = raw['sel']
        if match != sel and match.replace('*', '') not in sel: raw = {}
    if 'ecm' in raw:
        if raw['ecm'] != ecm: raw = {}
        else: raw.pop('ecm')

    if 'format' in raw and 'file_formats' in parameters:
        raw['file_formats'] = raw['format']
    if 'file_formats' in raw and 'format' in parameters:
        raw['format'] = raw['file_formats']

    result = defaults.copy()
    result.update({name: raw[name]     for name in defaults   if name in raw})
    result.update({name: context[name] for name in parameters if name in context})
    if 'ecm'       in parameters: result['ecm']       = ecm
    if 'lumi'      in parameters: result['lumi']      = lumi
    if 'selection' in parameters: result['selection'] = sel
    if 'variable'  in parameters: result['variable']  = var
    if function is makePlot and 'sig_scale' not in raw:
        result['sig_scale'] = 1. if context.get('cat') in ('ee', 'mumu') else 10.
    return result



######################
### MAIN FUNCTIONS ###
######################

def significance(
    variable: str,
    inputdir: str,
    outputdir: str,
    selection: str,
    plots: dict[str, dict[str, list[str]]],
    var_labels: dict[str, str],
    locx: str = 'right',
    locy: str = 'top',
    xMin: Union[float, int, None] = None,
    xMax: Union[float, int, None] = None,
    outName: str = '',
    suffix: str = '',
    file_formats: list[str] = ['png'],
    reverse: bool = False,
    lazy: bool = True,
    rebin: int = 1
     ) -> None:
    '''Plot running significance and signal efficiency for cut optimization.

    Scans cumulative signal/background yields across histogram bins, computes
    significance (S/√(S+B)), and displays both significance and signal efficiency
    on dual Y-axes with matplotlib. Marks the optimal cut point visually.

    Cumulative Direction:
    - reverse=False: S(>x) and B(>x) — left-to-right cumulative
    - reverse=True: S(<x) and B(<x) — right-to-left cumulative

    Args:
        variable (str): Name of the variable to optimize.
        inDir (str): Path to input directory containing histograms.
        outDir (str): Path to output directory for saving plots.
        sel (str): Selection tag for output organization.
        procs (list[str]): Process names; first is signal, rest are backgrounds.
        processes (dict[str, list[str]]): Mapping from process names to sample identifiers.
        locx (str, optional): Legend horizontal position ('left'/'right'). Defaults to 'right'.
        locy (str, optional): Legend vertical position ('top'/'bottom'). Defaults to 'top'.
        xMin (float | int | None, optional): Variable range lower bound. Defaults to None.
        xMax (float | int | None, optional): Variable range upper bound. Defaults to None.
        outName (str, optional): Base name for output file (default: variable). Defaults to ''.
        suffix (str, optional): Suffix to append to filename. Defaults to ''.
        format (list[str], optional): Image formats ('png', 'pdf', etc.). Defaults to ['png'].
        reverse (bool, optional): If True, compute right-to-left cumulative. Defaults to False.
        lazy (bool, optional): Use lazy loading for histograms. Defaults to True.
        rebin (int, optional): Rebinning factor for histograms. Defaults to 1.
    '''

    import numpy as np
    from plot.pyplot import MatplotlibPlot
    from tools.process import getHist
    from utilities import label_parts

    # Checking only 1 signal process is in plots
    if len(plots['signals']) != 1:
        raise ValueError('Only support one signal process')

    # Getting signals and backgrounds processes
    sig_procs = plots['signals'][next(iter(plots['signals']))]
    bkg_procs = [proc for bkg in plots['backgrounds'].values() for proc in bkg]

    # Getting signal and backgrounds histograms
    h_sig = getHist(variable, sig_procs, inputdir, f'_{selection}_histo', rebin, lazy)
    h_bkg = getHist(variable, bkg_procs, inputdir, f'_{selection}_histo', rebin, lazy)

    # Converting the histograms to numpy arrays
    nbins = h_sig.GetNbinsX()
    sig_arr = np.array(h_sig, dtype=np.float64)[1:nbins+1]
    bkg_arr = np.array(h_bkg, dtype=np.float64)[1:nbins+1]

    # Defining bin centers

    centers = h_sig.GetXaxis().GetXmin() + (np.arange(nbins) + 0.5) * \
        (h_sig.GetXaxis().GetXmax() - h_sig.GetXaxis().GetXmin()) / nbins

    # Mask to remove bins outside the allowed range
    mask = np.ones(nbins, dtype=bool)
    if xMin is not None: mask &= (centers >= xMin)
    if xMax is not None: mask &= (centers <= xMax)
    if not np.any(mask):
        raise ValueError('No cut values remain after applying xMin/xMax')

    # Compute cumulative sums from either left or right depending on reverse flag.
    tot_arr = sig_arr + bkg_arr
    sig_cum = np.cumsum(sig_arr) if reverse else np.cumsum(sig_arr[::-1])[::-1]
    tot_cum = np.cumsum(tot_arr) if reverse else np.cumsum(tot_arr[::-1])[::-1]

    significance = np.divide(sig_cum, np.sqrt(tot_cum), out=np.zeros_like(sig_cum), where=tot_cum > 0)
    sig_loss     = np.divide(sig_cum, h_sig.Integral(), out=np.zeros_like(sig_cum), where=h_sig.Integral() > 0,)

    x, y, l = centers[mask], significance[mask], sig_loss[mask]
    max_idx = int(np.argmax(y))
    max_x, max_y, max_l = float(x[max_idx]), float(y[max_idx]), float(l[max_idx])

    plotname, plot_suffix = variable if not outName else outName, f'{suffix}_reverse' if reverse else suffix
    main_plot = MatplotlibPlot(outputdir, 'significance', selection, plotname, plot_suffix, file_formats)
    effs_plot = MatplotlibPlot(axes=main_plot.axes.twinx(), style=False)

    try:
        main_plot.draw([
            ([x, y, 'scatter'], {'color': 'blue', 'marker': '.', 'label': 'Significance'}),
            ([max_x, max_y, 'scatter'], {'color': 'red', 'marker': '*', 's': 150}),
            ([max_x, 'axvline'], {'color': 'black', 'alpha': 0.8, 'linewidth': 1}),
            ([max_y, 'axhline'], {'color': 'blue', 'alpha': 0.8, 'linewidth': 1}),
        ])
        effs_plot.draw([
            ([x, l, 'plot'], {'color': 'red', 'linewidth': 3, 'label': 'Signal efficiency'}),
            ([max_l, 'axhline'], {'color': 'red', 'alpha': 0.8, 'linewidth': 1}),
        ])

        main_plot.configure(xlim=(min(x), max(x)))
        label, unit = label_parts(var_labels[variable])

        main_plot.set_labels(var_labels[variable], 'Significance', left=' ', locx=locx, locy=locy)
        main_plot.set_axis_color('y', 'blue')

        effs_plot.set_labels(ylabel='Signal Efficiency', left=' ', locy=locy)
        effs_plot.set_axis_color('y', 'red')
        effs_plot.axes.grid(False, axis='y')

        title = rf'{label} ${"<" if reverse else ">"}$ {max_x:.2f} {unit}, Z = {max_y:.2f}, $\epsilon_{{sig}} = {max_l*100:.1f}\%$'
        main_plot.set_title(title)
        main_plot.save(close=False)

    finally:
        main_plot.close()

    return None


def makePlot(
        variable: str,
        inputdir: str,
        outputdir: str,
        selection: str,
        plots: dict[str, dict[str, list[str]]],
        ecm: int = 240,
        lumi: float = 10.8,
        xmin: float | int | None = None,
        xmax: float | int | None = None,
        ymin: float | int | None = None,
        ymax: float | int | None = None,
        logX: bool = False,
        logY: bool = False,
        xtitle: str = '',
        ytitle: str = 'Events',
        xlabels: list[str] = [],
        outName: str = '',
        suffix: str = '',
        sig_scale: float = 1.,
        bkg_scale: float = 1.,
        scale_min: float | None = None,
        scale_max: float | None = None,
        rebin: int = 1,
        file_formats: list[str] = ['png'],
        stack: bool = False,
        strict: bool = True,
        lazy: bool = True,
        tot: bool = False,
        quiet: bool = True
)-> None:

    from plot.histoplot import HistogramPlot
    from constants import colors, legend

    histoplot = HistogramPlot(variable, selection, inputdir, outputdir, 'makePlot',
                              plots, colors, legend, ecm, lumi, tot)

    legend = histoplot.define_legend(len(plots['signals']) + len(plots['backgrounds']))
    all_hists = histoplot.load_histograms(f'_{selection}_histo', rebin, lazy)
    sig_hists, bkg_hists = histoplot.style_histograms(all_hists, sig_scale, bkg_scale)

    histoplot.build_config(sig_hists, bkg_hists, xmin, xmax, ymin, ymax,
                           logX, logY, xtitle, ytitle, scale_min, scale_max, strict, stack)

    canvas = histoplot.draw(all_hists, bkg_hists, stack, xlabels)

    outName = variable if not outName else outName
    histoplot.save(canvas, outName, suffix, logY, file_formats, quiet)
    canvas.Close()

    return None


def PlotDecays(
    variable: str,
    inputdir: str,
    outputdir: str,
    selection: str,
    plots: dict[str, dict[str, list[str]]],
    ecm: int = 240,
    lumi: float = 10.8,
    xmin: Union[float, int, None] = None,
    xmax: Union[float, int, None] = None,
    ymin: Union[float, int, None] = None,
    ymax: Union[float, int, None] = None,
    logX: bool = False,
    logY: bool = False,
    xtitle: str = '',
    ytitle: str = 'Unit Area',
    xlabels: list[str] = [],
    outName: str = '',
    suffix: str = '',
    scale_min: float | None = None,
    scale_max: float | None = None,
    rebin: int = 1,
    file_formats: list[str] = ['png'],
    strict: bool = True,
    lazy: bool = True,
    tot: bool = False,
    quiet: bool = True
     ) -> None:
    '''Plot Higgs decay modes with unit-integral normalization for shape comparison.

    Creates overlaid histograms for each Higgs decay channel, each normalized to
    unity to enable direct shape comparison across decay modes. Combines Z decay
    channels (ee, mumu, tautau) for each Higgs decay.

    Args:
        variable (str): Variable to plot.
        inDir (str): Path to input histogram files.
        outDir (str): Path for output plots.
        sel (str): Selection tag for organization.
        z_decays (list[str]): Z boson decay modes (e.g., ['ee', 'mumu', 'tautau']).
        h_decays (list[str]): Higgs decay modes to compare (e.g., ['bb', 'WW', 'ZZ']).
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        rebin (int, optional): Histogram rebinning factor. Defaults to 1.
        outName (str, optional): Base output filename (default: variable). Defaults to ''.
        suffix (str, optional): Filename suffix. Defaults to ''.
        format (list[str], optional): Image formats ['png', 'pdf']. Defaults to ['png'].
        xmin (float | int | None, optional): X-axis lower limit. Defaults to None.
        xmax (float | int | None, optional): X-axis upper limit. Defaults to None.
        ymin (float | int | None, optional): Y-axis lower limit. Defaults to None.
        ymax (float | int | None, optional): Y-axis upper limit. Defaults to None.
        logX (bool, optional): Use logarithmic X-axis. Defaults to False.
        logY (bool, optional): Use logarithmic Y-axis. Defaults to False.
        lazy (bool, optional): Use lazy histogram loading. Defaults to True.
        strict (bool, optional): Strict axis range enforcement. Defaults to True.
        tot (bool, optional): Save to 'tot' subdir if True, 'cat' subdir if False. Defaults to False.
    '''

    from plot.histoplot import HistogramPlot
    from constants import h_colors as colors, h_labels as legend

    histoplot = HistogramPlot(variable, selection, inputdir, outputdir, 'higgsDecays',
                              plots, colors, legend, ecm, lumi, tot)

    histoplot.define_legend(len(plots['signals']), 4,
                            0.2, 0.925, 0.95, 0.925)
    all_hists = histoplot.load_histograms(f'_{selection}_histo', rebin, lazy, True)
    sig_hists, _ = histoplot.style_histograms(all_hists)

    histoplot.build_config(sig_hists, [], xmin, xmax, ymin, ymax,
                           logX, logY, xtitle, ytitle, scale_min, scale_max, strict)

    canvas = histoplot.draw(all_hists, [], False, xlabels)

    outName = variable if not outName else outName
    histoplot.save(canvas, outName, suffix, logY, file_formats, quiet)
    canvas.Close()

    return None


def AAAyields(
    hName: str,
    inputdir: str,
    outputdir: str,
    selection: str,
    plots: dict[str, list[str]],
    cat: str,
    ecm: int = 240,
    lumi: float = 10.8,
    sig_scale: float = 1.,
    bkg_scale: float = 1.,
    suffix: str = '',
    lazy: bool = True,
    tot: bool = False,
    outName: str = '',
    file_formats: list[str] = ['png'],
    quiet: bool = False
     ) -> None:
    '''Render a yields summary canvas with process list and metadata.

    Creates a ROOT canvas displaying process yields, scaling factors, significance,
    and analysis metadata as formatted LaTeX text. Useful for publications.

    Args:
        hName (str): Histogram name/key to extract yields from.
        inDir (str): Path to input histogram files.
        outDir (str): Path for output plots.
        plots (dict[str, list[str]]): Dictionary with 'signal' and 'backgrounds' keys, each mapping to process lists.
        legend (dict[str, str]): Process name to display label mapping.
        colors (dict[str, str]): Process name to fill color mapping.
        cat (str): Category ('ee' or 'mumu') for analysis channel label.
        sel (str): Selection criteria identifier.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        scale_sig (float, optional): Scale factor for signal yields. Defaults to 1.0.
        scale_bkg (float, optional): Scale factor for background yields. Defaults to 1.0.
        lazy (bool, optional): Use lazy histogram loading. Defaults to True.
        tot (bool, optional): Include all the Z decays. Defaults to False.
        outName (str, optional): Base output filename. Defaults to 'AAAyields'.
        format (list[str], optional): Image formats. Defaults to ['png'].

    Raises:
        ValueError: If cat is not 'ee' or 'mumu'.
    '''

    if outName=='': outName = 'AAAyields'
    if   cat == 'mumu':
        ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow #mu^{+}#mu^{-} + X'
    elif cat == 'ee':
        ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow e^{+}e^{-} + X'
    elif cat == 'qq':
        ana_tex = 'e^{+}e^{-} #rightarrow ZH #rightarrow q#bar{q} + X'
    else:
        raise ValueError(f'{cat} value is not supported')

    import numpy as np
    from constants import colors, legend
    from plot.textplot import TextPlot

    textplot = TextPlot(inputdir, outputdir, 'yield', selection, ecm, lumi, tot)
    nprocs = len([*plots.get('signals', {}), *plots.get('backgrounds', {})])
    textplot.define_legend(nprocs)

    rows, s_tot, b_tot = textplot.load_yields(hName, plots, colors, legend, sig_scale, bkg_scale, lazy)
    with np.errstate(divide='ignore', invalid='ignore'):
        significance = s_tot / (s_tot + b_tot)**0.5 if s_tot > 0 and b_tot > 0 else 0

    metadata = [
        ('#bf{FCC-ee} #scale[0.7]{#it{Simulation}}', 0.05, 0.95, 0.05),
        (f'#it{{#sqrt{{s}} = {ecm} GeV}}', 0.065, 0.88, 0.04),
        (f'#it{{L = {lumi} ab^{{#minus1}}}}', 0.065, 0.83, 0.035),
        (f'#it{{{ana_tex}}}', 0.065, 0.78, 0.04),
        (f'#it{{{selection}}}', 0.065, 0.73, 0.025),
        (f'#it{{Signal Scaling = {sig_scale:.3g}}}', 0.065, 0.67, 0.04),
        (f'#it{{Background Scaling = {bkg_scale:.3g}}}', 0.065, 0.62, 0.04),
        (f'#it{{Significance = {significance:.3f}}}', 0.065, 0.57, 0.04),
    ]
    # Label, integral, entries
    rows = [(l, f'{i:,.0f}', f'{e:,.0f}') for l, i, e in rows]
    canvas = textplot.draw(rows, metadata)

    textplot.save(canvas, outName, suffix, file_formats, quiet)

    return None


def get_efficiency(
        hName: str,
        inputdir: str,
        ecm: int,
        z_decays: list[str],
        h_decays: list[str],
        suffix: str = '',
        invert: bool = False
         ):

    import os, uproot
    from tools.process import getMetaInfo, getHist

    sig_groups = ([[f'wzp6_ee_{z}H_H{h}_ecm{ecm}' for h in h_decays] for z in z_decays] if invert else
                  [[f'wzp6_ee_{z}H_H{h}_ecm{ecm}' for z in z_decays] for h in h_decays])
    lumi = {240: 10.8e6, 365: 3.12e6}.get(ecm, -1)
    effs, errs = {}, {}

    for decay, sigs in zip(h_decays, sig_groups):
        processed = sum(uproot.open(f'{inputdir}/{sig}{suffix}.root')['eventsProcessed'].value
                        for sig in sigs if os.path.exists(f'{inputdir}/{sig}{suffix}.root'))
        total = sum(getMetaInfo(signal, rmww=True) for signal in sigs) * lumi
        tot_err = total / processed**0.5

        histogram = getHist(hName, sigs, inputdir, suffix)
        sel, entries = histogram.Integral(), histogram.GetEntries()
        sel_err = entries**0.5 * total / processed
        eff = 100 * sel / total

        coeff = ((tot_err / total)**2 + ((sel_err / sel)**2 if sel > 0 else 0))**0.5

        effs[decay] = eff
        errs[decay] = eff * coeff

    return effs, errs


def Efficiency(
    hName: str,
    inputdir: str,
    outputdir: str,
    selection: str,
    z_decays: list[str],
    h_decays: list[str],
    h_labels: dict[str, str],
    suffix: str = '',
    outName: str = 'selection_efficiency',
    file_formats: list[str] = ['png'],
    ecm: int = 240,
    invert: bool = False,
    quiet: bool = False
     ) -> None:

    '''Generate efficiency summary plots and detailed comparison tables.

    Creates a pull-plot canvas showing final-step efficiencies for each decay mode
    relative to the average, with uncertainty error bars. Overlays average efficiency
    line and uncertainty band. Exports both plot and per-cut efficiency table.

    Plot Elements:
    - Y-axis: Decay channels + average
    - X-axis: Efficiency percentage at final cut
    - Markers: Final efficiency per decay ± uncertainty
    - Vertical line: Average efficiency
    - Shaded band: ±1σ uncertainty around average

    Args:
        outDir (str): Output directory for plots and tables.
        h_decays (list[str]): Ordered list of decay channel identifiers.
        suffix (str, optional): String appended to output file names. Defaults to ''.
        format (list[str], optional): Image formats (e.g., ['png', 'pdf']). Defaults to ['png'].
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
    '''

    from plot.pullplot import PullPlot

    lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

    efficiency, efficiency_err = get_efficiency(hName, inputdir, ecm, z_decays, h_decays,
                                                f'_{selection}_histo', invert)
    eff, eff_err = list(efficiency.values()), list(efficiency_err.values())
    eff_avg      = sum(eff) / len(eff)
    eff_avg_err  = (sum(err**2 for err in eff_err))**0.5 / len(eff_err)
    eff_min, eff_max = eff_avg - min(eff), max(eff) - eff_avg

    decays = z_decays if invert else h_decays
    plot = PullPlot(outputdir, 'yield', selection, eff, eff_err,
                    [h_labels[decay] for decay in decays], eff_avg, eff_avg_err,
                    ecm, lumi, 'Selection efficiency [%]', 'efficiency')
    plot.draw(outName, suffix, file_formats, quiet,
              top_right=f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}',
              statistics=(f'Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %',
                          f'Min/max: {eff_min:.2f}/{eff_max:.2f}'))

    plot.save_text(outName, 'Eff', '.2f', '.2f')

    return None


def CutFlow(
    flow: dict[str, dict[str, Any | dict[str, float]]],
    outputdir: str,
    selection: str,
    plots: dict[str, dict[str, dict[str, list[str]]]],
    colors: dict[str, dict[str, int]],
    legend: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    ecm: int = 240,
    lumi: float = 10.8,
    file_formats: list[str] = ['png'],
    outName: str = 'cutFlow',
    suffix: str = '',
    sig_scale: float = 1.0,
    yMin: float | None = 1e2,
    yMax: float | None = 1e12,
    tot: bool = False,
    quiet: bool = False,
) -> None:
    '''Render cutflow histogram with stacked backgrounds, signal overlay, and significance.

    Produces a stacked histogram showing event yields across sequential cut steps,
    with optional signal scaling. Overlays a histogram outline for total background
    and marks signal with scaled line style. Computes and exports significance
    (S/√(S+B)) and yields with Poisson uncertainties.

    Drawing Order:
    1. Frame (dummy histogram for axis setup)
    2. Background stack (filled)
    3. Total background outline (black line)
    4. Signal histogram (scaled, line style)
    5. Legend

    Yields Table Columns: Cut, Significance, Process1_yield±error, Process2_yield±error, ...

    Args:
        flow (dict[str, dict[str, ROOT.TH1 | dict[str, float]]]): Histograms and metadata indexed by process name.
        outDir (str): Base directory for output plots and tables.
        cat (str): Detector channel ('ee' for electron or 'mumu' for muon).
        sel (str): Selection identifier for retrieving cut definitions and labels.
        procs (list[str]): Process names in order [signal, background1, background2, ...].
        colors (dict[str, dict[str, ROOT.TColor]]): ROOT color mappings nested by channel and process.
        legend (dict[str, dict[str, str]]): Human-readable legend labels nested by channel and process.
        cuts (dict[str, dict[str, str]]): Cut expression definitions per selection (sel -> dict[cut_name -> expression]).
        labels (dict[str, dict[str, str]]): Axis labels per cut step (sel -> dict[cut_index -> label]).
        ecm (int, optional): Beam energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        outName (str, optional): Output file stem. Defaults to 'cutFlow'.
        format (list[str], optional): Image formats (e.g., ['png', 'pdf']). Defaults to ['png'].
        suffix (str, optional): String appended to file names. Defaults to ''.
        sig_scale (float, optional): Multiplicative factor for signal visibility. Defaults to 1.0.
        yMin (float, optional): Log-scale Y-axis minimum. Defaults to 1e4.
        yMax (float, optional): Log-scale Y-axis maximum. Defaults to 1e10.
    '''

    from plot.cutflowplot import CutFlowPlot

    procs = list(plots.get('signals', {})) + list(plots.get('backgrounds', {}))

    cutflow = CutFlowPlot(flow, outputdir, selection, plots, colors, legend, ecm, lumi, tot)
    cutflow.prepare(sig_scale)
    cutflow.define_legend(sig_scale, len(procs))
    rows = cutflow.draw(labels[selection], yMin, yMax, outName, file_formats, suffix, quiet)

    # Export yields table to file
    cutflow.write_table(outName+suffix, ['Cut', 'Significance'] + procs, rows)

    return None


def CutFlowDecays(
    flow: dict[str, dict[str, Any | dict[str, float]]],
    outputdir: str,
    selection: str,
    labels: dict[str, dict[str, str]],
    ecm: int = 240,
    lumi: float = 10.8,
    file_formats: list[str] = ['png'],
    outName: str = 'cutFlow_decays',
    suffix: str = '',
    yMin: float | int = 0,
    yMax: float | int = 150,
    tot: bool = False,
    quiet: bool = False
) -> None:
    '''Plot selection efficiencies across Higgs decay modes as normalized curves.

    Renders efficiency curves (normalized to first cut as 100%) for each Higgs decay
    channel overlaid on a single plot. Computes average efficiency, spreads (min/max),
    and generates detailed tables of efficiency values and uncertainties.

    Drawing Elements:
    - Efficiency curves per decay mode (colored lines)
    - Average efficiency line (gray)
    - Uncertainty band around average (shaded region)
    - Statistics box with average ± uncertainty and min/max spreads

    Args:
        flow (dict[str, dict[str, ROOT.TH1 | dict[str, float]]]): Histograms indexed by decay channel.
        outDir (str): Base directory for outputs (plots and tables).
        cat (str): Detector channel ('ee' for electron or 'mumu' for muon).
        sel (str): Selection identifier for cut definitions and labels.
        labels (dict[str, dict[str, str]]): Axis labels corresponding to each cut step.
        suffix (str, optional): String appended to file names. Defaults to ''.
        ecm (int, optional): Beam energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        outName (str, optional): Output file stem. Defaults to 'cutFlow_decays'.
        format (list[str], optional): Image formats (e.g., ['png', 'pdf']). Defaults to ['png'].
        yMin (float | int, optional): Linear Y-axis minimum (efficiency %). Defaults to 0.
        yMax (float | int, optional): Linear Y-axis maximum (efficiency %). Defaults to 150.
    '''

    from plot.cutflowplot import CutFlowPlot
    from constants import h_colors
    h_decays = list(flow)

    # Let CutFlowPlot convert the flow dictionary and prepare normalized curves.
    plots = {'signals': {decay: [] for decay in h_decays}, 'backgrounds': {}}
    flow_plot = CutFlowPlot(flow, outputdir, selection, plots,
                            h_colors, h_labels, ecm, lumi, tot)
    flow_plot.prepare(rescale=True)
    hists = flow_plot.signals
    if not hists:
        raise ValueError('CutFlowDecays requires at least one decay histogram')
    nbins = next(iter(hists)).GetNbinsX()

    # Extract final bin efficiency and uncertainty
    final = [float(h.GetBinContent(nbins)) for h in hists]

    # Compute average efficiency across decay channels
    hist_tot = None
    for hist in hists:
        if hist_tot is None: hist_tot = hist.Clone('h_tot')
        else: hist_tot.Add(hist)
    hist_tot.Scale(1.0 / len(h_decays))
    
    avg     = hist_tot.GetBinContent(nbins)
    avg_err = hist_tot.GetBinError(nbins)
    
    # Min/max spreads relative to average
    Min, Max = avg-min(final), max(final)-avg

    flow_plot.define_legend(1, len(h_decays))
    labels = [labels[selection][key] for key in sorted(labels[selection])]
    rows = flow_plot.draw(labels, yMin, yMax, outName, file_formats, suffix, quiet, (avg, avg_err, Min, Max))
    flow_plot.write_table(outName+suffix, ['Cut', *h_decays], rows, 10, 25)

    return None
