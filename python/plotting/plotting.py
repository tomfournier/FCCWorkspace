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
from re import search
from typing import Any, Union, TYPE_CHECKING

from ..tools.process import getHist

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import ROOT

from constants import h_labels
from tools.utils import mkdir
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

def _label_parts(label: str) -> tuple[str, str]:
    match = search(r'\s*\[([^]]+)\]\s*$', label)
    return (label[:match.start()], match.group(1)) if match else (label, '')

def _ensure_plt_style() -> None:
    '''Initialize matplotlib styling once to avoid repeated setup calls.

    Sets the global PLT_STYLE_SET flag after first initialization to prevent
    redundant style configuration in subsequent calls.
    '''
    global PLT_STYLE_SET
    if not PLT_STYLE_SET:
        from .python.plotter import set_plt_style
        set_plt_style()
        PLT_STYLE_SET = True

def _parse_selection_dir(
    sel: str,
    outDir: str,
    subdir: str
     ) -> str:
    '''Build standardized output directory path from selection string.

    Returns a path of the form:
        outDir/subdir/<base_sel>/<direction>

    Where:
    - base_sel: selection name without suffixes ('_high', '_low')
    - direction: one of 'high', 'low', or 'nominal'
    '''
    base_sel = sel.replace('_high', '').replace('_low', '')
    direction = 'high' if '_high' in sel else ('low' if '_low' in sel else 'nominal')
    return f'{outDir}/{subdir}/{base_sel}/{direction}'

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
    return {'signals': {decay: get_process_dict([process], ecm, h_decays=[decay])[process]
                        for decay in H_decays}}


def plot_configs(ecm: int, cat: str) -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    from samples import get_process_dict
    backgrounds = get_process_dict(['WW', 'ZZ', 'Zgamma', 'Rare'] +
                                   (['tt'] if cat == 'qq' and ecm == 365 else []), ecm)
    category = {'signals': get_process_dict([f'Z{cat}H'], ecm),
                'backgrounds': backgrounds}
    total = {'signals': get_process_dict(['ZH'], ecm),
             'backgrounds': backgrounds}
    return {
        'category': category,
        'total':    total,
        'decay':       decay_plots(f'Z{cat}H', ecm),
        'total_decay': decay_plots('ZH', ecm),
    }



######################
### MAIN FUNCTIONS ###
######################

# ___________________________________________
def get_args(
    var: str,
    function,
    args: dict[str, dict[str, Union[str, float, int]]],
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
    sel, ecm, lumi = context['sel'], context['ecm'], context['lumi']
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
    if 'ecm'      in parameters: result['ecm']      = ecm
    if 'lumi'     in parameters: result['lumi']     = lumi
    if 'sel'      in parameters: result['sel']      = sel
    if 'variable' in parameters: result['variable'] = var
    if function is makePlot and 'sig_scale' not in raw:
        result['sig_scale'] = 1. if context.get('cat') in ('ee', 'mumu') else 10.
    return result


# ________________________________________
def significance(
    variable: str,
    inDir: str,
    outDir: str,
    sel: str,
    plots: dict[str, dict[str, list[str]]],
    var_labels: dict[str, str],
    locx: str = 'right',
    locy: str = 'top',
    xMin: Union[float, int, None] = None,
    xMax: Union[float, int, None] = None,
    outName: str = '',
    suffix: str = '',
    format: list[str] = ['png'],
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
    import matplotlib.pyplot as plt
    from .python.plotter import set_labels, savefigs
    from ..tools.process import getHist

    _ensure_plt_style()


    if outName=='': outName = variable
    suff  = f'_{sel}_histo'

    sig_procs = list(plots['signals'])
    if len(sig_procs) != 1:
        raise ValueError('Only support one signal process')
    sig = sig_procs[0]
    h_sig = getHist(variable,
                    plots['signals'][sig], inDir,
                    suffix=suff, rebin=rebin)
    sig_tot = h_sig.Integral()

    bkgs_procs = []
    for bkg in plots['backgrounds'].keys():
        bkgs_procs.extend(plots['backgrounds'][bkg])

    h_bkg = getHist(variable, bkgs_procs, inDir,
                    suffix=suff, rebin=rebin, lazy=lazy)

    nbins = h_sig.GetNbinsX()

    sig_arr = np.array(h_sig, dtype=np.float64)[:nbins+1]
    bkg_arr = np.array(h_bkg, dtype=np.float64)[:nbins+1]

    # Get axis object once
    xaxis = h_sig.GetXaxis()

    # Check if variable bin width
    if xaxis.IsVariableBinSize():
        # Variable bins: extract edges individually but efficiently
        centers = np.array([xaxis.GetBinCenter(i+1) for i in range(nbins+1)], dtype=np.float64)
    else:
        # Fixed bins: use linspace for maximum speed
        centers = np.linspace(xaxis.GetXmin(), xaxis.GetXmax(), nbins + 1, dtype=np.float64)

    mask = np.ones(nbins+1, dtype=bool)
    if xMin is not None: mask &= (centers >= xMin)
    if xMax is not None: mask &= (centers <= xMax)

    # Compute cumulative sums from either left or right depending on reverse flag.
    if reverse:
        sig_cum = np.cumsum(sig_arr)
        bkg_cum = np.cumsum(bkg_arr)
    else:
        # Right-to-left cumulative: sum from high to low bin values.
        sig_cum = np.cumsum(sig_arr[::-1])[::-1]
        bkg_cum = np.cumsum(bkg_arr[::-1])[::-1]

    denom = sig_cum + bkg_cum
    with np.errstate(divide='ignore', invalid='ignore'):
        significance = np.where(denom > 0, sig_cum / np.sqrt(denom), 0)
    sig_loss = np.where(sig_tot > 0, sig_cum / sig_tot, 0)

    x, y = centers[mask], significance[mask]
    l = sig_loss[mask]

    max_index = int(np.argmax(y))
    max_y = float(y[max_index])
    max_x, max_l = float(x[max_index]), float(l[max_index])

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax2.plot(x, l, color='red', linewidth=3,
             label='Signal efficiency')
    ax1.scatter(x, y, color='blue', marker='o',
                label='Significance')
    ax1.scatter(max_x, max_y, color='red',
                marker='*', s=150)

    ax1.axvline(max_x, color='black', alpha=0.8, linewidth=1)
    ax1.axhline(max_y, color='blue',  alpha=0.8, linewidth=1)
    ax2.axhline(max_l, color='red',   alpha=0.8, linewidth=1)

    ax1.set_xlim(min(x), max(x))
    label, unit = _label_parts(var_labels[variable])

    set_labels(ax1, var_labels[variable], 'Significance', left=' ', locx=locx, locy=locy)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.yaxis.label.set_color('blue')

    set_labels(ax2, ylabel='Signal Efficiency', left=' ', locy=locy)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.yaxis.label.set_color('red')
    ax2.grid(False, axis='y')

    if reverse:
        ax1.set_title(rf'Max: {label} $<$ {max_x:.2f} {unit}, '
                      rf'Significance = {max_y:.2f}, Signal eff = {max_l*100:.1f} \%')
    else:
        ax1.set_title(rf'Max: {label} $>$ {max_x:.2f} {unit}, '
                      rf'Significance = {max_y:.2f}, Signal eff = {max_l*100:.1f} \%')
    fig.tight_layout()

    out = _parse_selection_dir(sel, outDir, 'significance')
    mkdir(out)

    suffix = '_reverse' if reverse else ''
    savefigs(fig, out, outName, suffix, format)
    plt.close()


def makePlot(
        variable: str,
        inDir: str,
        outDir: str,
        sel: str,
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

    from plots.histoplot import HistogramPlot
    from constants import colors, legend

    histoplot = HistogramPlot(variable, sel, inDir, outDir,
                              plots, colors, legend, ecm, lumi, tot)

    legend = histoplot.define_legend(len(plots['signals']) + len(plots['backgrounds']))
    all_hists = histoplot.load_histograms(f'_{sel}_histo', rebin, lazy)
    stack_hist, sig_hists, bkg_hists = histoplot.style_histograms(
        all_hists, legend, sig_scale, bkg_scale)

    histoplot.cfg = histoplot.build_config(sig_hists, bkg_hists,
                                           xmin, xmax, ymin, ymax, logX, logY,
                                           xtitle, ytitle, scale_min, scale_max, strict, stack)

    canvas, _ = histoplot.draw(all_hists, stack_hist, bkg_hists,
                               legend, stack, xlabels)

    outName = variable if not outName else outName
    histoplot.save(canvas, outName, suffix, file_formats, logY, quiet)
    canvas.Close()


def PlotDecays(
    variable: str,
    inDir: str,
    outDir: str,
    sel: str,
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

    from plots.histoplot import HistogramPlot
    from constants import h_colors as colors, h_labels as legend

    histoplot = HistogramPlot(variable, sel, inDir, outDir,
                              plots, colors, legend, ecm, lumi, tot)

    legend = histoplot.define_legend(len(plots['signals']), 4,
                                     0.2, 0.925, 0.95, 0.925)
    all_hists = histoplot.load_histograms(f'_{sel}_histo', rebin, lazy, True)
    _, hists, _ = histoplot.style_histograms(all_hists, legend)

    histoplot.cfg = histoplot.build_config(hists, [],
                                           xmin, xmax, ymin, ymax, logX, logY,
                                           xtitle, ytitle, scale_min, scale_max, strict)

    canvas, _ = histoplot.draw(all_hists, None, [],
                               legend, False, xlabels)

    outName = variable if not outName else outName
    histoplot.save(canvas, outName, suffix, file_formats, logY, quiet)
    canvas.Close()


# _______________________________
def AAAyields(
    hName: str,
    inDir: str,
    outDir: str,
    plots: dict[str, list[str]],
    cat: str, sel: str,
    ecm: int = 240,
    lumi: float = 10.8,
    scale_sig: float = 1.,
    scale_bkg: float = 1.,
    lazy: bool = True,
    tot: bool = False,
    outName: str = '',
    format: list[str] = ['png'],
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
    from plots.textplot import TextPlot

    textplot = TextPlot(outDir, sel, ecm=ecm, lumi=lumi)
    rows, leg, s_tot, b_tot = textplot.load_yields(
        hName, inDir, plots, legend, colors,
        scale_sig, scale_bkg, lazy
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        significance = s_tot / (s_tot + b_tot)**0.5 if s_tot > 0 and b_tot > 0 else 0

    metadata = [
        ('#bf{FCC-ee} #scale[0.7]{#it{Simulation}}', 0.9, 0.92, 0.04),
        (f'#bf{{#it{{#sqrt{{s}} = {ecm} GeV}}}}', 0.18, 0.83, 0.04),
        (f'#bf{{#it{{L = {lumi} ab^{{#minus1}}}}}}', 0.18, 0.78, 0.035),
        (f'#bf{{#it{{{ana_tex}}}}}', 0.18, 0.73, 0.04),
        (f'#bf{{#it{{{sel}}}}}', 0.18, 0.68, 0.025),
        (f'#bf{{#it{{Signal Scaling = {scale_sig:.3g}}}}}', 0.18, 0.62, 0.04),
        (f'#bf{{#it{{Background Scaling = {scale_bkg:.3g}}}}}', 0.18, 0.57, 0.04),
        (f'#bf{{#it{{Significance = {significance:.3f}}}}}', 0.18, 0.52, 0.04),
    ]
    formatted_rows = [
        (label, f'{integral:,.0f}', f'{entries:,.0f}')
        for label, integral, entries in rows
    ]
    textplot.draw(formatted_rows, metadata, outName, legend=leg,
                  file_formats=format, quiet=quiet,
                  suffix='_tot' if tot else '')


def get_efficiency(
        hName: str,
        inDir: str,
        ecm: int,
        z_decays: list[str],
        h_decays: list[str],
        suffix: str = '',
        invert: bool = False
         ):

    import os, uproot
    from tools.process import getMetaInfo

    signal_groups = ([[f'wzp6_ee_{z}H_H{h}_ecm{ecm}' for h in h_decays] for z in z_decays] if invert else
                     [[f'wzp6_ee_{z}H_H{h}_ecm{ecm}' for z in z_decays] for h in h_decays])
    lumi = {240: 10.8e6, 365: 3.12e6}.get(ecm, -1)
    efficiencies, uncertainties = {}, {}

    for decay, signals in zip(h_decays, signal_groups):
        processed = sum(
            uproot.open(f'{inDir}/{signal}{suffix}.root')['eventsProcessed'].value
            for signal in signals if os.path.exists(f'{inDir}/{signal}{suffix}.root'))
        total = sum(getMetaInfo(signal, rmww=True) for signal in signals) * lumi
        total_error = total / processed**0.5

        histogram = getHist(hName, signals, inDir, suffix)
        selected, entries = histogram.Integral(), histogram.GetEntries()
        selected_error = entries**0.5 * total / processed
        efficiency = 100 * selected / total

        efficiencies[decay]  = efficiency
        uncertainties[decay] = efficiency * (
            (total_error / total)**2 + ((selected_error / selected)**2
                                        if selected > 0 else 0))**0.5

    return efficiencies, uncertainties


def Efficiency(
    hName: str,
    inDir: str,
    outDir: str,
    sel: str,
    z_decays: list[str],
    h_decays: list[str],
    h_labels: dict[str, str],
    suffix: str = '',
    outName: str = 'selection_efficiency',
    format: list[str] = ['png'],
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

    from plots.pullplot import PullPlot

    lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

    efficiency, efficiency_err = get_efficiency(hName, inDir, ecm, z_decays, h_decays, f'_{sel}_histo', invert)
    eff, eff_err = list(efficiency.values()), list(efficiency_err.values())
    eff_avg     = sum(eff) / len(eff)
    eff_avg_err = (sum(err**2 for err in eff_err))**0.5 / len(eff_err)
    eff_min, eff_max = eff_avg - min(eff), max(eff) - eff_avg

    decays = z_decays if invert else h_decays
    plot = PullPlot(eff, eff_err, [h_labels[decay] for decay in decays],
                    eff_avg, eff_avg_err, ecm, lumi,
                    'Selection efficiency [%]', out_dir=outDir, selection=sel)
    plot.draw(None, outName, suffix, format, quiet,
              top_right=f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}',
              statistics=(f'Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %',
                          f'Min/max: {eff_min:.2f}/{eff_max:.2f}'))

    plot.save_text(plot.output_dir(), outName,
                   'Eff', '.2f', '.2f')

    return None
