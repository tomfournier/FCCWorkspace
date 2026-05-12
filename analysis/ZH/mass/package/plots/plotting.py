'''Plotting utilities.

Provides:
- Plotting functions for histograms and distributions: `makePlot()`, `PlotDecays()`,
  `significance()`, `PseudoRatio()`, `AAAyields()`, `Bias()`.
- Utility functions for managing plot arguments: `get_args()`, `args_decay()`,
  `_ensure_plt_style()`.
- Histogram conversion utilities: `hist_to_arrays()`.
- Integration with ROOT graphics backend for publication-quality plots.
- Support for both ROOT and matplotlib visualization backends.

Functions:
- `significance()`: Plot running significance and signal efficiency for cut optimization.
- `makePlot()`: Draw signal/background histograms with optional stacking.
- `PlotDecays()`: Compare Higgs decay modes with unit normalization.
- `PseudoRatio()`: Create ratio plots comparing nominal and pseudo-signal distributions.
- `AAAyields()`: Render yields summary canvas with process yields and metadata.
- `Bias()`: Plot bias distributions per Higgs decay mode with uncertainty bands.
- `get_args()`, `args_decay()`: Manage and merge plotting configuration arguments.

Conventions:
- Uses ROOT TLatex for axis labels and LaTeX text rendering.
- Supports both linear and logarithmic scaling on x and y axes.
- Integrates with `config.py` for color palettes, labels, and variable definitions.
- Output plots saved to hierarchical subdirectories by selection and category.

Usage:
- Create publication plots with automatic styling and legend management.
- Plot signal significance as function of cut value for optimization.
- Compare decay mode distributions across Higgs and Z decay channels.

Lazy Imports:
- numpy and pandas are lazy-loaded only when their specific functions are called
- ROOT is lazy-loaded via local imports in functions (Python caches the module automatically)
- ROOT-dependent helpers are lazy-loaded via local imports
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from ..config import (
    vars_label,
    vars_xlabel
)
from ..tools.utils import mkdir
from ..logger import get_logger

LOGGER = get_logger(__name__)



########################
### CONFIG AND SETUP ###
########################

# Tracks whether the matplotlib style has been set for this session.
PLT_STYLE_SET = False



########################
### HELPER FUNCTIONS ###
########################

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
    var_args: dict,
    ecm: int,
    sel: str
     ) -> dict[str,
               Union[float,
                     int,
                     str]]:
    '''Navigate nested args structure to extract parameters for given ecm and sel.

    Supports three levels of nesting:
    1. args[var] = {...parameters...}  # Direct parameters
    2. args[var][ecm] = {...parameters...}  # Organized by ecm
    3. args[var][ecm][sel_pattern] = {...parameters...}  # Organized by ecm and sel

    Also supports:
    - args[var][sel_pattern] = {...parameters...}  # Organized by sel only

    Args:
        var_args (dict): The args dictionary for a specific variable.
        ecm (int): Center-of-mass energy in GeV.
        sel (str): Selection criteria identifier.

    Returns:
        dict: Dictionary of parameters, or empty dict if no match found.
    '''

    current = var_args

    # Try to navigate by ecm (as integer key) if present
    if ecm in current:
        current = current[ecm]

    # Try to find matching sel pattern in current level
    # Look for string keys that could be selection patterns
    param_keys = {
        'xmin', 'xmax', 'ymin', 'ymax', 'rebin', 'which', 'sel',
        'ecm', 'lumi', 'suffix', 'outName', 'format', 'strict',
        'logX', 'logY', 'stack', 'sig_scale', 'lazy', 'tot'
    }

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



######################
### MAIN FUNCTIONS ###
######################

# ___________________________________________
def get_args(
    var: str,
    sel: str,
    ecm: int,
    lumi: float,
    args: dict[str,
               dict[str,
                    Union[str, float, int]]]
     ) -> dict[str,
               Union[str, float, int]]:
    '''Return plotting arguments for a variable/selection pair with defaults.

    Copies user-provided options for a given variable, applies selection filters,
    resolves the `which` flag, and fills missing keys with sensible defaults.

    Supports hierarchical args structure:
    - args[var] = {...params...}
    - args[var][ecm] = {...params...}
    - args[var][ecm][sel_pattern] = {...params...}
    - args[var][sel_pattern] = {...params...}

    Args:
        var (str): Variable name to retrieve arguments for.
        sel (str): Selection criteria identifier.
        ecm (int): Center-of-mass energy in GeV.
        lumi (float): Integrated luminosity in ab^-1.
        args (dict[str, dict[str, Union[str, float, int]]]): Nested dictionary of plotting arguments indexed by variable name.

    Returns:
        dict[str, Union[str, float, int]]: Dictionary of plotting arguments with all required keys populated.
    '''

    # Use helper function to navigate nested structure
    arg = _extract_nested_args(args[var], ecm, sel) if var in args else {}

    if 'which' in arg:
        if arg['which']=='both':
            del arg['which']
        elif arg['which']=='make':
            del arg['which']
        elif arg['which']=='decay':
            arg = {}
        else:
            LOGGER.warning("Wrong value given to 'which', acting as if 'both' were given")


    if 'sel' in arg:
        if '*' in arg['sel']:
            if arg['sel'].replace('*', '') not in sel:
                arg = {}
            else:
                del arg['sel']
        elif arg['sel']!=sel:
            arg = {}
        else:
            del arg['sel']

    if 'ecm' in arg and arg['ecm']==ecm:
        del arg['ecm']
    elif 'ecm' in arg and arg['ecm']!=ecm:
        arg = {}
    else:
        pass

    for key in ['xmin', 'xmax', 'ymin', 'ymax']:
        arg.setdefault(key, None)

    arg.setdefault('rebin', 1)
    arg.setdefault('ecm',  ecm)
    arg.setdefault('lumi', lumi)

    arg.setdefault('suffix',  '')
    arg.setdefault('outName', '')
    arg.setdefault('format', ['png'])

    arg.setdefault('strict', True)
    arg.setdefault('logX',   False)

    arg.setdefault('stack',  False)
    arg.setdefault('sig_scale', 1.)

    return arg


# ________________________________________
def significance(
    variable: str,
    inDir: str,
    outDir: str,
    sel: str,
    procs: list[str],
    processes: dict[str, list[str]],
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
    '''Plot running significance and signal efficiency for a cut variable.

    Builds cumulative signal/background yields across histogram bins, computes
    significance (s/sqrt(s+b)), and displays the optimal cut point on a dual-axis
    matplotlib figure with significance and signal efficiency curves.

    Args:
        variable (str): Name of the variable to optimize.
        inDir (str): Path to input directory containing histograms.
        outDir (str): Path to output directory for saving plots.
        sel (str): Selection tag for output organization.
        procs (list[str]): Process names; first is signal, rest are backgrounds.
        processes (dict[str, list[str]]): Mapping from process names to file/sample identifiers.
        locx (str, optional): Legend horizontal position ('left', 'right'). Defaults to 'right'.
        locy (str, optional): Legend vertical position ('top', 'bottom'). Defaults to 'top'.
        xMin (float | int | None, optional): Range limits for variable. Defaults to None.
        xMax (float | int | None, optional): Range limits for variable. Defaults to None.
        outName (str, optional): Base name for output file. Defaults to variable name.
        suffix (str, optional): Suffix to append to filename. Defaults to ''.
        format (list[str], optional): Image formats to save ('png', 'pdf', etc.). Defaults to ['png'].
        reverse (bool, optional): If True, compute left-to-right cumulative (v > x). Defaults to False.
        lazy (bool, optional): Use lazy loading for histograms. Defaults to True.
        rebin (int, optional): Rebinning factor. Defaults to 1.
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from .python.plotter import set_labels, savefigs
    from ..tools.process import getHist

    _ensure_plt_style()


    if outName=='': outName = variable
    suff  = f'_{sel}_histo'

    h_sig = getHist(variable,
                    processes[procs[0]], inDir,
                    suffix=suff, rebin=rebin)
    sig_tot = h_sig.Integral()

    bkgs_procs = []
    for bkg in procs[1:]:
        bkgs_procs.extend(processes[bkg])

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
    if xMin is not None:
        mask &= (centers >= xMin)
    if xMax is not None:
        mask &= (centers <= xMax)

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
    ax2.plot(
        x, l, color='red',
        linewidth=3,
        label='Signal efficiency'
    )
    ax1.scatter(
        x, y, color='blue',
        marker='o',
        label='Significance'
    )
    ax1.scatter(
        max_x, max_y, color='red',
        marker='*', s=150
    )

    ax1.axvline(
        max_x, color='black',
        alpha=0.8, linewidth=1
    )
    ax1.axhline(
        max_y, color='blue',
        alpha=0.8, linewidth=1
    )
    ax2.axhline(
        max_l, color='red',
        alpha=0.8, linewidth=1
    )

    ax1.set_xlim(min(x), max(x))
    if variable=='H':
        GeV = ' GeV$^{2}$'
    elif 'GeV' in vars_xlabel[variable]:
        GeV = ' GeV'
    else:
        GeV = ''

    set_labels(ax1, vars_xlabel[variable], 'Significance', left=' ', locx=locx, locy=locy)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.yaxis.label.set_color('blue')

    set_labels(ax2, ylabel='Signal Efficiency', left=' ', locy=locy)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.yaxis.label.set_color('red')
    ax2.grid(False, axis='y')

    if reverse:
        ax1.set_title(
            rf'Max: {vars_label[variable]} $<$ {max_x:.2f}{GeV}, '
            rf'Significance = {max_y:.2f}, '
            rf'Signal eff = {max_l*100:.1f} \%'
        )
    else:
        ax1.set_title(
            rf'Max: {vars_label[variable]} $>$ {max_x:.2f}{GeV}, '
            rf'Significance = {max_y:.2f}, '
            rf'Signal eff = {max_l*100:.1f} \%'
        )
    fig.tight_layout()

    s = sel.replace('_high', '').replace('_low', '')
    if '_high' in sel: d = 'high'
    elif '_low' in sel: d = 'low'
    else: d = 'nominal'
    out = f'{outDir}/significance/{s}/{d}'
    mkdir(out)

    suffix = '_reverse' if reverse else ''
    savefigs(
        fig, out, outName,
        suffix=suffix,
        format=format
    )
    plt.close()

# ________________________________________
def makePlot(
    variable: str,
    inDir: str,
    outDir: str,
    sel: str,
    procs: list[str],
    processes: dict[str, list[str]],
    colors: dict[str, str],
    legend: dict[str, str],
    ecm: int = 240,
    lumi: float = 10.8,
    suffix: str = '',
    outName: str = '',
    format: list[str] = ['png'],
    xmin: Union[float, int, None] = None,
    xmax: Union[float, int, None] = None,
    ymin: Union[float, int, None] = None,
    ymax: Union[float, int, None] = None,
    rebin: int = 1,
    sig_scale: float = 1.,
    strict: bool = True,
    logX: bool = False,
    logY: bool = True,
    stack: bool = False,
    lazy: bool = True
     ) -> None:
    '''Draw signal/background histogram with optional stacking.

    Loads and styles histograms, applies rebinning and scaling, and renders
    with configurable axis ranges, log scales, and stacking options.

    Args:
        variable (str): Variable name to plot.
        inDir (str): Path to input histogram files.
        outDir (str): Path for output plots.
        sel (str): Selection tag for organization.
        procs (list[str]): Process names; first is signal, rest are backgrounds.
        processes (dict[str, list[str]]): Process name to sample/file mapping.
        colors (dict[str, str]): Process name to color mapping (ROOT color codes).
        legend (dict[str, str]): Process name to legend label mapping.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        suffix (str, optional): Filename suffix. Defaults to ''.
        outName (str, optional): Base output filename. Defaults to variable name.
        format (list[str], optional): Image formats. Defaults to ['png'].
        xmin (float | int | None, optional): Axis range limits. Defaults to None.
        xmax (float | int | None, optional): Axis range limits. Defaults to None.
        ymin (float | int | None, optional): Axis range limits. Defaults to None.
        ymax (float | int | None, optional): Axis range limits. Defaults to None.
        rebin (int, optional): Rebinning factor. Defaults to 1.
        sig_scale (float, optional): Signal histogram scale factor. Defaults to 1.0.
        strict (bool, optional): Strict axis range enforcement. Defaults to True.
        logX (bool, optional): Use logarithmic scale. Defaults to False.
        logY (bool, optional): Use logarithmic scale. Defaults to True.
        stack (bool, optional): Stack backgrounds. Defaults to False.
        lazy (bool, optional): Use lazy histogram loading. Defaults to True.
    '''

    # Lazy-load ROOT and helpers
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    from .root import plotter
    from .root.plotter import finalize_canvas
    from .root.helper import (
        mk_legend, load_hists, build_cfg, style_hist, save_plot
    )

    if outName == '': outName = variable
    suff = f'_{sel}_histo'

    Processes = {k:v for k, v in processes.items() if k in procs}

    leg = mk_legend(len(procs))
    raw_hists = load_hists(
        Processes,
        variable,
        inDir,
        suffix=suff,
        rebin=rebin,
        lazy=lazy
    )

    # Extract signal histogram and initialize background stack.
    sig_key = procs[0]
    sig_hist = raw_hists.get(sig_key)

    st, bkgs = ROOT.THStack(), []
    st.SetName('stack')

    # Style histograms in-place without cloning for faster execution
    for proc in procs:
        hist = raw_hists.get(proc)
        if not hist:
            continue

        is_sig = proc == sig_key
        scale = f' (#times {int(sig_scale)})' if is_sig and sig_scale!=1 else ''
        style_hist(
            hist,
            color=colors[proc] if is_sig else ROOT.kBlack,
            width=3 if is_sig else 1,
            fill_color=colors[proc] if not is_sig else None,
            scale=sig_scale if is_sig else 1.
        )
        leg.AddEntry(hist, legend[proc]+scale, 'L' if is_sig else 'F')

        if not is_sig:
            st.Add(hist)
            bkgs.append(hist)

    cfg = build_cfg(
        sig_hist,
        logX=logX, logY=logY,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        ecm=ecm, lumi=lumi,
        strict=strict,
        stack=stack,
        hists=bkgs
    )

    plotter.cfg = cfg
    canvas, dummy = plotter.canvas(), plotter.dummy()
    dummy.Draw('HIST')
    if stack:
        st.Add(sig_hist)
        st.Draw('HIST SAME')
    else:
        if bkgs:
            st.Draw('HIST SAME')
        sig_hist.Draw('HIST SAME')
    leg.Draw('SAME')

    finalize_canvas(canvas)
    out = _parse_selection_dir(sel, outDir, 'makePlot')
    linlog = '_log' if logY else '_lin'
    save_plot(canvas, out, outName, linlog+suffix, format)
    # Explicitly delete objects to free memory faster
    canvas.Close()
    del canvas, dummy, leg, st


# _______________________________
def AAAyields(
    hName: str,
    inDir: str,
    outDir: str,
    plots: dict[str, list[str]],
    legend: dict[str, str],
    colors: dict[str, str],
    cat: str, sel: str,
    ecm: int = 240,
    lumi: float = 10.8,
    scale_sig: float = 1.,
    scale_bkg: float = 1.,
    lazy: bool = True,
    outName: str = '',
    format: list[str] = ['png']
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
    else:
        raise ValueError(f'{cat} value is not supported')

    # Lazy-load ROOT, numpy and helpers
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    from .root import plotter
    from ..tools.process import getHist
    from .root.helper import (
        mk_legend, style_hist, savecanvas, draw_latex, configure_axis
    )

    suffix = f'_{sel}_histo'

    signal      = plots['signal']
    backgrounds = plots['backgrounds']

    leg = mk_legend(
        len(signal)+len(backgrounds),
        x1=0.6, y1=0.86, x2=0.9, y2=0.88,
        text_font=42)

    yields = {}
    for s in signal:
        hist = getHist(
            hName,
            signal[s], inDir,
            suffix=suffix, lazy=lazy,
            use_cache=False
        )
        integral = hist.Integral()
        entries  = hist.GetEntries()

        style_hist(
            hist,
            color=colors[s],
            scale=scale_sig,
            width=4
        )
        leg.AddEntry(hist, legend[s], 'L')
        yields[s] = [
            legend[s],
            integral * scale_sig,
            entries
        ]

    for b in backgrounds:
        hist = getHist(
            hName,
            backgrounds[b], inDir,
            suffix=suffix, lazy=lazy,
            use_cache=False
        )
        if hist is None:
            LOGGER.warning(f"Couldn't find histograms for {b}")
        integral = hist.Integral()
        entries  = hist.GetEntries()

        style_hist(
            hist,
            color=ROOT.kBlack,
            fill_color=colors[b],
            scale=scale_bkg
        )
        leg.AddEntry(hist, legend[b], 'F')

        yields[b] = [
            legend[b],
            integral * scale_bkg,
            entries
        ]

    canvas = plotter.canvas(
        top=None, bottom=None,
        left=0.14, right=0.08,
        batch=True, yields=True
    )

    dummyh = ROOT.TH1F('', '', 1, 0, 1)
    dummyh.SetStats(0)
    configure_axis(
        dummyh.GetXaxis(),
        '', 0, 1,
        label_offset=999,
        label_size=0
    )
    configure_axis(
        dummyh.GetYaxis(),
        '', 0, 1,
        label_offset=999,
        label_size=0
    )
    dummyh.Draw('AH')
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(31)

    simu = [('#bf{FCC-ee} #scale[0.7]{#it{Simulation}}', 0.9, 0.92, 0.04)]
    draw_latex(latex, simu)


    latex.SetTextAlign(12)
    latex.SetNDC(ROOT.kTRUE)
    sqrt = [('#bf{#it{#sqrt{s} = '+f'{ecm}'+' GeV}}', 0.18, 0.83, 0.04)]
    draw_latex(latex, sqrt)

    # Compute total signal and background for significance calculation.
    s_tot = int(sum([yields[s][1] for s in signal]))
    b_tot = int(sum([yields[b][1] for b in backgrounds]))
    # Compile analysis metadata for LaTeX rendering.
    with np.errstate(divide='ignore', invalid='ignore'):
        z = s_tot/(s_tot+b_tot)**(0.5) if s_tot>0 and b_tot>0 else 0
    text_data = [
        ('#bf{#it{L = '+f'{lumi}'+' ab^{#minus1}}}', 0.18, 0.78, 0.035),
        ('#bf{#it{' + ana_tex + '}}', 0.18, 0.73, 0.04),
        ('#bf{#it{' + sel + '}}', 0.18, 0.68, 0.025),
        ('#bf{#it{Signal Scaling = ' + f'{scale_sig:.3g}'
            '}}', 0.18, 0.62, 0.04),
        ('#bf{#it{Background Scaling = '
         f'{scale_bkg:.3g}' + '}}', 0.18, 0.57, 0.04),
        ('#bf{#it{Significance = ' +
         f'{z:.3f}' + '}}', 0.18, 0.52, 0.04),
        ('#bf{#it{Process}}', 0.18, 0.45, 0.035),
        ('#bf{#it{Yields}}', 0.5, 0.45, 0.035),
        ('#bf{#it{Raw MC}}', 0.75, 0.45, 0.035),
    ]
    draw_latex(latex, text_data)

    latex_yield = []
    for dy, (label, integral, entries) in enumerate(yields.values()):
        latex_yield.extend([
            ('#bf{#it{' + label + '}}', 0.18, 0.4-dy*0.05, 0.035),
            ('#bf{#it{' + str(int(integral)) + '}}', 0.5, 0.4-dy*0.05, 0.035),
            ('#bf{#it{' + str(int(entries)) + '}}', 0.75, 0.4-dy*0.05, 0.035)
        ])
    draw_latex(latex, latex_yield)

    out = _parse_selection_dir(sel, outDir, 'yield')
    mkdir(out)
    savecanvas(canvas, out, outName, format=format)
    canvas.Close()
