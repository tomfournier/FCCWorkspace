'''Pure-Python plotting utilities (uproot + hist + mplhep + matplotlib).

Refactor goals:
- Remove ROOT graphics; keep look-and-feel with mplhep/matplotlib.
- Read histograms via uproot with minimal I/O and memory copies.
- Provide compact, fast implementations; co-locate hot helpers here to
  reduce cross-file imports and conversion overhead.

Notes:
- We keep function signatures to minimize call-site changes.
- We avoid importing heavy ROOT modules here. Config still provides labels,
  which use TLatex-like tokens; we translate them to LaTeX for matplotlib.
- Colors passed as hex/Matplotlib names work out-of-the-box.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Union, Iterable, Tuple, Dict, List

import uproot
import mplhep as hep
import hist
from matplotlib.pyplot import subplots, close

from ..config import (
    h_colors,
    h_labels,
    vars_label,
    vars_xlabel,
)
from ..tools.utils import mkdir
from .python.plotter import set_plt_style, set_labels, savefigs



########################
### CONFIG AND SETUP ###
########################

PLT_STYLE_SET = False

def _ensure_plt_style() -> None:
    '''Initialize mpl style once (kept minimal to avoid overhead).'''
    global PLT_STYLE_SET
    if not PLT_STYLE_SET:
        hep.style.use('CMS')  # close to previous styling
        set_plt_style()
        PLT_STYLE_SET = True


########################
### LIGHT HELPERS    ###
########################

# Minimal palette fallback for decay labels if config uses ROOT ints
_DECAY_PALETTE = {
    'bb': 'tab:purple', 'cc': 'tab:blue', 'ss': 'tab:red',
    'gg': 'tab:green', 'mumu': 'tab:orange', 'tautau': 'tab:cyan',
    'ZZ': 'gray', 'WW': 'dimgray', 'Za': 'tab:olive', 'aa': 'tab:red',
    'inv': 'tab:blue'
}

def _color_safe(val, key: str | None = None, default: str = 'tab:blue') -> str:
    if isinstance(val, str):
        return val
    if key and key in _DECAY_PALETTE:
        return _DECAY_PALETTE[key]
    return default

def _tlx_to_tex(s: str) -> str:
    '''Translate common ROOT TLatex tokens to matplotlib LaTeX.
    Keeps it simple and fast; cover tokens used in config labels.'''
    rep = {
        '#rightarrow': r'\rightarrow',
        '#leftarrow': r'\leftarrow',
        '#mu': r'\mu',
        '#tau': r'\tau',
        '#gamma': r'\gamma',
        '#nu': r'\nu',
        '#pm': r'\pm',
        '#pi': r'\pi',
        '#bf{': r'\\mathbf{',
        '#it{': r'\\mathit{',
        '#bar{': r'\\bar{',
        '#minus': '-',
        '#plus': '+',
        '#sqrt{': r'\\sqrt{',
        ' }': ' ',
    }
    out = s
    for k, v in rep.items():
        out = out.replace(k, v)
    return f'${out}$' if '$' not in out else out


def _to_hist(hobj,
             rebin: int = 1
             ) -> hist.Hist:
    '''Convert uproot histo to hist.Hist with Weight storage and optional rebin.'''
    try:
        h = hobj.to_hist()
    except Exception:
        vals, edges = hobj.to_numpy(flow=False)
        vars_p = hobj.variances()
        h = hist.Hist(hist.axis.Variable(edges), storage=hist.storage.Weight())
        view = h.view()
        view.value = vals.astype(float)
        view.variance = (vars_p.astype(float)
                         if vars_p is not None else vals.astype(float))
    if rebin != 1:
        h = h[::hist.rebin(rebin)]
    return h

def _hist_arrays(h: hist.Hist
                 ) -> Tuple[np.ndarray, 
                            np.ndarray, 
                            np.ndarray]:
    vals = h.values(flow=False)
    edges = h.axes[0].edges
    err = np.sqrt(h.variances(flow=False))
    return vals, edges, err

def _read_hist_sum(hname: str,
                   procs: Iterable[str],
                   in_dir: str,
                   suffix: str = '',
                   rebin: int = 1,
                   lazy: bool = True) -> hist.Hist | None:
    '''Fast uproot reader: build a summed hist.Hist with propagated errors.'''
    h_sum: hist.Hist | None = None
    for p in procs:
        fpath = os.path.join(in_dir, f'{p}{suffix}.root')
        if not os.path.exists(fpath):
            if lazy:
                continue
            raise FileNotFoundError(f'Input file not found: {fpath}')
        try:
            with uproot.open(fpath) as f:
                hobj = f[hname]
                h = _to_hist(hobj, rebin=rebin)
                h_sum = h if h_sum is None else (h_sum + h)
        except Exception:
            if not lazy:
                raise
            continue
    return h_sum


def _stack_ymax(bkgs: List[np.ndarray], sig: np.ndarray | None) -> float:
    if not bkgs and sig is None:
        return 1.0
    if not bkgs:
        return float(np.max(sig)) if sig is not None else 1.0
    bsum = np.sum(np.vstack(bkgs), axis=0)
    if sig is not None:
        return float(np.max(bsum + sig))
    return float(np.max(bsum))


def _range_from_arrays(edges: np.ndarray,
                       sigs: List[np.ndarray],
                       bkgs: List[np.ndarray],
                       logY: bool = False,
                       strict: bool = True,
                       stack: bool = False,
                       xmin: float | int | None = None,
                       xmax: float | int | None = None,
                       ymin: float | int | None = None,
                       ymax: float | int | None = None) -> Tuple[float, float, float, float]:
    # X-range: restrict to bins with content if strict
    contents = np.sum(np.vstack(sigs + bkgs), axis=0) if (sigs or bkgs) else np.array([0])
    mask = contents > 0 if strict else np.ones_like(contents, dtype=bool)
    x_edges = edges
    if mask.any():
        i = np.argmax(mask)
        j = len(mask) - np.argmax(mask[::-1])
        x_min, x_max = x_edges[i], x_edges[j]
    else:
        x_min, x_max = x_edges[0], x_edges[-1]
    if xmin is not None: x_min = max(x_min, xmin)
    if xmax is not None: x_max = min(x_max, xmax)

    # Y-range
    if stack:
        ymax_auto = _stack_ymax(bkgs, sigs[0] if sigs else None)
    else:
        ymax_auto = max([np.max(a) for a in (sigs + bkgs)] or [1.0])
    if logY:
        positives = np.concatenate([a[a > 0] for a in (sigs + bkgs)])
        ymin_auto = float(np.min(positives)) if positives.size else 1e-3
        ymin_auto *= 0.5
    else:
        ymin_auto = 0.0
    y_min = ymin if ymin is not None else ymin_auto
    y_max = ymax if ymax is not None else float(ymax_auto * (10.0 if logY else 1.4))
    return float(x_min), float(x_max), float(y_min), float(y_max)


######################
### MAIN FUNCTIONS ###
######################

#____________________________________________________
def get_args(var: str,
             sel: str,
             ecm: int,
             lumi: float,
             args: Dict[str, Dict[str, Union[str, float, int]]]
             ) -> Dict[str, Union[str, float, int]]:
    '''Return plotting arguments for a variable/selection pair with defaults.'''
    arg = args[var].copy() if var in args else {}
    if 'which' in arg:
        if arg['which'] in ('both', 'make'): arg.pop('which', None)
        elif arg['which'] == 'decay': arg = {}
        else: print("WARNING: wrong 'which'; using 'both'")
    if 'sel' in arg:
        m = arg['sel']
        if '*' in m:
            if m.replace('*', '') not in sel: arg = {}
            else: arg.pop('sel', None)
        elif m != sel: arg = {}
        else: arg.pop('sel', None)
    if 'ecm' in arg:
        if arg['ecm'] == ecm: arg.pop('ecm', None)
        else: arg = {}
    for k in ('xmin', 'xmax', 'ymin', 'ymax'): arg.setdefault(k, None)
    arg.setdefault('rebin', 1)
    arg.setdefault('ecm', ecm)
    arg.setdefault('lumi', lumi)
    arg.setdefault('suffix', '')
    arg.setdefault('outName', '')
    arg.setdefault('format', ['png'])
    arg.setdefault('strict', True)
    arg.setdefault('logX', False)
    arg.setdefault('stack', False)
    arg.setdefault('sig_scale', 1.0)
    return arg

#_____________________________________________________
def args_decay(var: str,
               sel: str,
               ecm: int,
               lumi: float,
               args: Dict[str, Dict[str, Union[str, float, int]]]
               ) -> Dict[str, Union[str, float, int]]:
    '''Return decay-plot arguments with defaults (exclude 'make' mode).'''
    arg = args[var].copy() if var in args else {}
    if 'which' in arg:
        w = arg['which']
        if w in ('both', 'decay'): arg.pop('which', None)
        elif w == 'make': arg = {}
        else: print("WARNING: wrong 'which'; using 'both'")
    if 'sel' in arg:
        m = arg['sel']
        if '*' in m:
            if m.replace('*', '') not in sel: arg = {}
            else: arg.pop('sel', None)
        elif m != sel: arg = {}
        else: arg.pop('sel', None)
    if 'ecm' in arg:
        if arg['ecm'] == ecm: arg.pop('ecm', None)
        else: arg = {}
    for k in ('xmin', 'xmax', 'ymin', 'ymax'): arg.setdefault(k, None)
    arg.setdefault('rebin', 1)
    arg.setdefault('ecm', ecm)
    arg.setdefault('lumi', lumi)
    arg.setdefault('suffix', '')
    arg.setdefault('outName', '')
    arg.setdefault('format', ['png'])
    arg.setdefault('strict', True)
    arg.setdefault('logX', False)
    return arg

#_____________________________________________________
def significance(variable: str,
                 inDir: str,
                 outDir: str,
                 sel: str,
                 procs: List[str],
                 processes: Dict[str, List[str]],
                 locx: str = 'right',
                 locy: str = 'top',
                 xMin: Union[float, int, None] = None,
                 xMax: Union[float, int, None] = None,
                 outName: str = '',
                 suffix: str = '',
                 format: List[str] = ['png'],
                 reverse: bool = False,
                 lazy: bool = True,
                 rebin: int = 1) -> None:
    '''Plot running significance and signal efficiency for a cut variable.'''
    _ensure_plt_style()
    if outName == '': outName = variable
    suff = f'_{sel}_histo'

    sig_hist = _read_hist_sum(variable,
                              processes[procs[0]],
                              inDir, suff, rebin, lazy)
    if sig_hist is None:
        return
    sig_vals, edges, _ = _hist_arrays(sig_hist)

    bkg_hists = []
    for b in procs[1:]:
        h = _read_hist_sum(variable, processes[b], inDir, suff, rebin, lazy)
        if h is not None:
            bkg_hists.append(h)
    if bkg_hists:
        bkg_sum = bkg_hists[0]
        for hb in bkg_hists[1:]:
            bkg_sum = bkg_sum + hb
        bkg_vals = bkg_sum.values(flow=False)
    else:
        bkg_vals = np.zeros_like(sig_vals)

    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = np.ones_like(sig_vals, dtype=bool)
    if xMin is not None: mask &= (centers >= xMin)
    if xMax is not None: mask &= (centers <= xMax)

    s = sig_vals
    b = bkg_vals
    if reverse:
        s_c = np.cumsum(s)
        b_c = np.cumsum(b)
    else:
        s_c = np.cumsum(s[::-1])[::-1]
        b_c = np.cumsum(b[::-1])[::-1]

    denom = s_c + b_c
    with np.errstate(divide='ignore', invalid='ignore'):
        sig = np.where(denom > 0, s_c / np.sqrt(denom), 0.0)
    eff = np.where(s.sum() > 0, s_c / s.sum(), 0.0)

    x = centers[mask]
    y = sig[mask]
    l = eff[mask]
    imax = int(np.argmax(y))
    y_max = float(y[imax])
    x_max, l_at = float(x[imax]), float(l[imax])

    fig, ax1 = subplots()
    ax2 = ax1.twinx()
    ax2.plot(x, l, color='red', lw=3, label='Signal efficiency')
    ax1.scatter(x, y, color='blue', s=18, label='Significance')
    ax1.scatter(x_max, y_max, color='red', marker='*', s=150)
    ax1.axvline(x_max, color='black', alpha=0.8, lw=1)
    ax1.axhline(y_max, color='blue', alpha=0.8, lw=1)
    ax2.axhline(l_at, color='red', alpha=0.8, lw=1)

    ax1.set_xlim(float(x.min()), float(x.max()))
    if variable == 'H':
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

    title = (rf'Max: {vars_label[variable]} $<$ {x_max:.2f}{GeV}, ' if reverse else
             rf'Max: {vars_label[variable]} $>$ {x_max:.2f}{GeV}, ')
    ax1.set_title(title + rf'Significance = {y_max:.2f}, Signal eff = {l_at*100:.1f} \%')
    fig.tight_layout()

    sel_clean = sel.replace('_high', '').replace('_low', '')
    d = 'high' if '_high' in sel else ('low' if '_low' in sel else 'nominal')
    out = f'{outDir}/significance/{sel_clean}/{d}'
    mkdir(out)
    sfx = '_reverse' if reverse else ''
    savefigs(fig, out, outName, suffix=sfx, format=format)
    close(fig)

#_________________________________________________
def makePlot(variable: str,
             inDir: str,
             outDir: str,
             sel: str,
             procs: List[str],
             processes: Dict[str, List[str]],
             colors: Dict[str, str],
             legend: Dict[str, str],
             ecm: int = 240,
             lumi: float = 10.8,
             suffix: str = '',
             outName: str = '',
             format: List[str] = ['png'],
             xmin: Union[float, int, None] = None,
             xmax: Union[float, int, None] = None,
             ymin: Union[float, int, None] = None,
             ymax: Union[float, int, None] = None,
             rebin: int = 1,
             sig_scale: float = 1.0,
             strict: bool = True,
             logX: bool = False,
             logY: bool = True,
             stack: bool = False,
             lazy: bool = True) -> None:
    '''Draw signal/background histograms with optional stacking (mplhep).'''
    _ensure_plt_style()
    if outName == '': outName = variable
    suff = f'_{sel}_histo'

    # Load histograms per process group (sum each group)
    hists: Dict[str, hist.Hist | None] = {}
    for k in procs:
        hists[k] = _read_hist_sum(variable, processes[k], inDir, suff, rebin, lazy)
    sig_key = procs[0]
    sig_hist = hists[sig_key]
    if sig_hist is None:
        return
    if sig_scale != 1.0:
        sig_hist = sig_hist * float(sig_scale)
    sig_vals, edges, _ = _hist_arrays(sig_hist)

    bkg_hists = []
    bkg_labels = []
    bkg_colors = []
    for k in procs[1:]:
        h = hists.get(k)
        if h is None:
            continue
        bkg_hists.append(h)
        bkg_labels.append(legend[k])
        bkg_colors.append(_color_safe(colors.get(k, 'gray'), default='gray'))

    bkg_arrays = [bh.values(flow=False) for bh in bkg_hists]
    xMin, xMax, yMin, yMax = _range_from_arrays(edges,
                                                [sig_vals],
                                                bkg_arrays,
                                                logY=logY,
                                                strict=strict,
                                                stack=stack,
                                                xmin=xmin, xmax=xmax,
                                                ymin=ymin, ymax=ymax)

    # Plot
    fig, ax = subplots()
    if logX: ax.set_xscale('log')
    if logY: ax.set_yscale('log')

    if bkg_hists:
        hep.histplot(bkg_hists, stack=True, histtype='fill',
                     color=bkg_colors, label=bkg_labels, ax=ax)
    hep.histplot(sig_hist, histtype='step', lw=3,
                 color=_color_safe(colors.get(sig_key, 'tab:blue'), default='tab:blue'),
                 label=(legend.get(sig_key, sig_key) +
                        (f' (x{int(sig_scale)})' if sig_scale != 1 else '')),
                 ax=ax)

    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    lumi_text = rf'$\sqrt{{s}} = {ecm}$ GeV, {lumi} ab$^{{-1}}$'
    set_labels(ax, vars_xlabel.get(variable, variable), 'Events', right=lumi_text)
    ax.legend(loc='best', frameon=True)

    sel_clean = sel.replace('_high', '').replace('_low', '')
    d = 'high' if '_high' in sel else ('low' if '_low' in sel else 'nominal')
    out = f'{outDir}/makePlot/{sel_clean}/{d}'
    mkdir(out)
    linlog = '_log' if logY else '_lin'
    savefigs(fig, out, outName or variable, linlog + suffix, format)
    close(fig)

#___________________________________________________
def PlotDecays(variable: str,
               inDir: str,
               outDir: str,
               sel: str,
               z_decays: List[str],
               h_decays: List[str],
               ecm: int = 240,
               lumi: float = 10.8,
               rebin: int = 1,
               outName: str = '',
               suffix: str = '',
               format: List[str] = ['png'],
               xmin: Union[float, int, None] = None,
               xmax: Union[float, int, None] = None,
               ymin: Union[float, int, None] = None,
               ymax: Union[float, int, None] = None,
               logX: bool = False,
               logY: bool = False,
               lazy: bool = True,
               strict: bool = True,
               tot: bool = False) -> None:
    '''Compare Higgs decay modes with unit normalization using mplhep.'''
    _ensure_plt_style()
    if outName == '': outName = variable
    suff = f'_{sel}_histo'

    # Build per-decay signal group names and load
    decay_hists: Dict[str, hist.Hist | None] = {}
    for h in h_decays:
        group = [f'wzp6_ee_{z}H_H{h}_ecm{ecm}' for z in z_decays]
        decay_hists[h] = _read_hist_sum(variable, group, inDir, suff, rebin, lazy)

    # Plot
    fig, ax = subplots()
    if logX: ax.set_xscale('log')
    if logY: ax.set_yscale('log')

    ref_edges = None
    line_arrays: List[np.ndarray] = []
    lines = []
    for h in h_decays:
        hsig = decay_hists.get(h)
        if hsig is None:
            continue
        total = float(hsig.values(flow=False).sum())
        scale = (1.0 / total) if total > 0 else 1.0
        hscaled = hsig * scale
        vals, edges, _ = _hist_arrays(hscaled)
        ref_edges = edges if ref_edges is None else ref_edges
        line_arrays.append(vals)
        color = _color_safe(h_colors.get(h, 'tab:blue'), key=h, default='tab:blue')
        label = _tlx_to_tex(h_labels.get(h, h))
        l = hep.histplot(hscaled, histtype='step', lw=2,
                         color=color, label=label, ax=ax)
        lines.append(l)

    if ref_edges is None or not line_arrays:
        return
    xMin, xMax, yMin, yMax = _range_from_arrays(ref_edges,
                                                line_arrays,
                                                [],
                                                logY=logY, strict=strict,
                                                xmin=xmin, xmax=xmax,
                                                ymin=ymin, ymax=ymax)
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    lumi_text = rf'$\sqrt{{s}} = {ecm}$ GeV, {lumi} ab$^{{-1}}$'
    set_labels(ax, vars_xlabel.get(variable, variable), 'Unit area', right=lumi_text)
    ax.legend(ncol=min(4, max(1, len(lines)//2)), loc='upper center')

    s = sel.replace('_high', '').replace('_low', '')
    d = 'high' if '_high' in sel else ('low' if '_low' in sel else 'nominal')
    base = f'{outDir}/higgsDecays/{s}/{d}/tot' if tot else f'{outDir}/higgsDecays/{s}/{d}/cat'
    mkdir(base)
    linlog = '_log' if logY else '_lin'
    savefigs(fig, base, outName or variable, linlog + suffix, format)
    close(fig)

#________________________________________
def AAAyields(hName: str,
              inDir: str,
              outDir: str,
              plots: Dict[str, List[str]],
              legend: Dict[str, str],
              colors: Dict[str, str],
              cat: str, sel: str,
              ecm: int = 240,
              lumi: float = 10.8,
              scale_sig: float = 1.0,
              scale_bkg: float = 1.0,
              lazy: bool = True,
              outName: str = '',
              format: List[str] = ['png']) -> None:
    '''Render a yields summary canvas using matplotlib text (ROOT-free).'''
    _ensure_plt_style()
    if outName == '': outName = 'AAAyields'
    if cat == 'mumu':
        ana_tex = r'$e^{+}e^{-} \to ZH \to \mu^{+}\mu^{-} + X$'
    elif cat == 'ee':
        ana_tex = r'$e^{+}e^{-} \to ZH \to e^{+}e^{-} + X$'
    else:
        raise ValueError(f'{cat} value is not supported')

    suffix = f'_{sel}_histo'
    signal = plots['signal']
    backgrounds = plots['backgrounds']

    # Collect yields via uproot; entries approximated as sum of bin contents.
    yields: Dict[str, Tuple[str, float, int]] = {}
    # Backgrounds
    for b in backgrounds:
        hb = _read_hist_sum(hName, backgrounds[b], inDir, suffix, 1, lazy)
        if hb is None:
            continue
        vals, _, _ = _hist_arrays(hb)
        y = float(np.sum(vals) * scale_bkg)
        n = int(np.sum(vals))
        yields[b] = (legend[b], y, n)
    # Signal(s)
    for s in signal:
        hs = _read_hist_sum(hName, signal[s], inDir, suffix, 1, lazy)
        if hs is None:
            continue
        vals, _, _ = _hist_arrays(hs)
        y = float(np.sum(vals) * scale_sig)
        n = int(np.sum(vals))
        yields[s] = (legend[s], y, n)

    # Totals and significance S/sqrt(S+B)
    s_tot = sum(y for k, (lbl, y, n) in yields.items() if k in signal)
    b_tot = sum(y for k, (lbl, y, n) in yields.items() if k in backgrounds)
    z = s_tot / np.sqrt(s_tot + b_tot) if (s_tot > 0 and b_tot > 0) else 0.0

    # Build figure with text blocks similar to ROOT canvas
    fig, ax = subplots()
    ax.axis('off')

    left = r'$\textbf{FCC-ee \textit{Simulation}}$'
    right = rf'$\sqrt{{s}} = {ecm}$ GeV'
    ax.set_title(left, loc='left')
    ax.set_title(right, loc='right')

    y0 = 0.85
    dy = 0.06
    fig.text(0.15, y0, rf'$L = {lumi}\,\mathrm{{ab^{{-1}}}}$', fontsize=22)
    fig.text(0.15, y0 - dy, ana_tex, fontsize=24)
    fig.text(0.15, y0 - 2*dy, rf'$\textit{{{sel}}}$', fontsize=18)
    fig.text(0.15, y0 - 3*dy, rf'$\textit{{Signal~Scaling}} = {scale_sig:.3g}$', fontsize=20)
    fig.text(0.15, y0 - 4*dy, rf'$\textit{{Background~Scaling}} = {scale_bkg:.3g}$', fontsize=20)
    fig.text(0.15, y0 - 5*dy, rf'$\textit{{Significance}} = {z:.3f}$', fontsize=20)

    # Table header
    fig.text(0.15, y0 - 6.5*dy, r'$\textbf{Process}$', fontsize=18)
    fig.text(0.50, y0 - 6.5*dy, r'$\textbf{Yields}$', fontsize=18)
    fig.text(0.75, y0 - 6.5*dy, r'$\textbf{Raw\ MC}$', fontsize=18)

    # Rows
    ystart = y0 - 7.2*dy
    for i, (key, (lbl, val, ent)) in enumerate(yields.items()):
        yline = ystart - i * (dy * 0.9)
        fig.text(0.15, yline, lbl, fontsize=16)
        fig.text(0.50, yline, f'{int(val)}', fontsize=16)
        fig.text(0.75, yline, f'{int(ent)}', fontsize=16)

    s = sel.replace('_high', '').replace('_low', '')
    d = 'high' if '_high' in sel else ('low' if '_low' in sel else 'nominal')
    out = f'{outDir}/yield/{s}/{d}'
    mkdir(out)
    savefigs(fig, out, outName, format=format)
    close(fig)

#___________________________________
def Bias(df: pd.DataFrame,
         nomDir: str,
         outDir: str,
         h_decays: List[str],
         suffix: str = '',
         ecm: int = 240,
         lumi: float = 10.8,
         outName: str = 'bias',
         format: List[str] = ['png']) -> None:
    '''Plot bias distribution per Higgs decay with uncertainty bands (matplotlib).'''
    _ensure_plt_style()
    bias_dict = dict(zip(df['mode'], df['bias'] * 100.0))
    unc = float(np.loadtxt(os.path.join(nomDir, 'results.txt'))[-1] * 1e4)

    vals = np.array([bias_dict[h] for h in h_decays])
    labels = [_tlx_to_tex(h_labels.get(h, h)) for h in h_decays]

    max_bias = np.max(np.abs(vals)) if vals.size else 1.0
    rng = max(unc, max_bias) * 1.2
    xMin, xMax = -rng, rng

    inside = np.abs(vals) < unc
    y = np.arange(len(h_decays)) + 0.5

    fig, ax = subplots()
    ax.set_ylim(0, len(h_decays))
    ax.set_xlim(xMin, xMax)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(True, axis='x', alpha=0.3)

    # Uncertainty band lines
    for x0, c in ((-unc, 'black'), (0.0, 'gray'), (unc, 'black')):
        ax.axvline(x0, color=c, lw=2)

    ax.scatter(vals[inside], y[inside], s=40, c='black', zorder=3)
    ax.scatter(vals[~inside], y[~inside], s=40, c='red', zorder=3)

    set_labels(ax, xlabel='Bias (×100) [%]', ylabel='')
    top = rf'$\sqrt{{s}} = {ecm}$ GeV, {lumi} ab$^{{-1}}$'
    ax.set_title(top, loc='right')
    ax.set_title(r'$\textbf{FCC-ee \textit{Simulation}}$', loc='left')

    mkdir(outDir)
    savefigs(fig, outDir, outName, suffix, format)
    close(fig)

#_________________________________________
def hist_to_arrays(obj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Convert uproot hist or (values, edges[, errors]) to arrays.'''
    if isinstance(obj, hist.Hist):
        return _hist_arrays(obj)
    if hasattr(obj, 'to_numpy'):
        v, e = obj.to_numpy(flow=False)
        var = obj.variances()
        err = np.sqrt(var) if var is not None else np.sqrt(v)
        return v, e, err
    if isinstance(obj, tuple):
        if len(obj) == 3: return obj
        if len(obj) == 2: return obj[0], obj[1], np.sqrt(obj[0])
    raise TypeError('Unsupported histogram object type')

#_____________________________________________
# Fast pseudo-signal builder (ROOT-free), used by PseudoRatio

def _signal_lists(cat: str,
                  z_decays: List[str],
                  h_decays: List[str],
                  target: str,
                  ecm: int = 240,
                  tot: bool = True) -> List[List[str]]:
    cats = z_decays if tot else [cat]
    tmpl = f'wzp6_ee_{{}}H_H{{}}_ecm{ecm}'
    h_list = [y.replace('ZZ', 'ZZ_noInv') if target == 'inv' else y for y in h_decays]
    return [[tmpl.format(z, h) for z in cats] for h in h_list]


def _make_pseudosignal_np(hName: str,
                          inDir: str,
                          target: str,
                          cat: str,
                          z_decays: List[str],
                          h_decays: List[str],
                          ecm: int = 240,
                          variation: float = 1.05,
                          suffix: str = '',
                          proc_scales: Dict[str, float] | None = None,
                          tot: bool = True,
                          rebin: int = 1,
                          lazy: bool = True) -> hist.Hist | None:
    if proc_scales is None: proc_scales = {}
    sigs = _signal_lists(cat, z_decays, h_decays, target, ecm=ecm, tot=tot)

    # Collect per-channel hists and totals
    hists: List[hist.Hist | None] = []
    totals: List[float] = []
    edges_ref = None
    zh_scale = float(proc_scales.get('ZH', 1.0))
    for sig in sigs:
        h = _read_hist_sum(hName, sig, inDir, suffix, rebin, lazy)
        if h is not None:
            h = h * zh_scale
            edges_ref = h.axes[0].edges if edges_ref is None else edges_ref
            totals.append(float(h.values(flow=False).sum()))
        else:
            totals.append(0.0)
        hists.append(h)
    if all(h is None for h in hists):
        return None

    S_tot = float(np.sum(totals))
    try:
        idx = h_decays.index(target)
    except ValueError:
        idx = -1
    S_target = float(totals[idx]) if idx >= 0 else 0.0
    scale_target = 1.0 + ((variation - 1.0) * (S_tot / S_target)) if S_target > 0 else 1.0

    # Build pseudo-signal: scale only target channel
    ps_hist: hist.Hist | None = None
    for i, h in enumerate(hists):
        if h is None:
            continue
        sc = scale_target if i == idx else 1.0
        h_scaled = h * sc
        ps_hist = h_scaled if ps_hist is None else (ps_hist + h_scaled)

    if ps_hist is None and edges_ref is not None:
        ps_hist = hist.Hist(hist.axis.Variable(edges_ref), storage=hist.storage.Weight())
    return ps_hist

#_____________________________________________
def PseudoRatio(variable: str,
                inDir: str,
                outDir: str,
                cat: str,
                target: str,
                z_decays: List[str],
                h_decays: List[str],
                ecm: int = 240,
                lumi: float = 10.8,
                pert: float = 1.05,
                sel: str = '',
                rebin: int = 1,
                outName: str = 'PseudoRatio',
                format: List[str] = ['png'],
                proc_scales: Dict[str, float] = {},
                logX: bool = False,
                logY: bool = False,
                lazy: bool = True,
                tot: bool = True,
                density: bool = False) -> None:
    '''Compare nominal signal sum to pseudo-signal; draw ratio with errors.'''
    _ensure_plt_style()
    if outName == '': outName = 'PseudoRatio'
    suffix = f'_{sel}_histo'

    # Sum all signal contributions into a single nominal histogram
    sigs = [[f'wzp6_ee_{z}H_H{h}_ecm{ecm}' for z in z_decays] for h in h_decays]
    sig_hist: hist.Hist | None = None
    for group in sigs:
        h = _read_hist_sum(variable, group, inDir, suffix, rebin, lazy)
        if h is None:
            continue
        sig_hist = h if sig_hist is None else (sig_hist + h)
    if sig_hist is None:
        return

    # Build pseudo-signal with target variation
    ps_hist = _make_pseudosignal_np(variable, inDir, target, cat,
                                    z_decays, h_decays, ecm=ecm,
                                    variation=pert, suffix=suffix,
                                    proc_scales=proc_scales,
                                    tot=tot, rebin=rebin, lazy=lazy)
    if ps_hist is None:
        return

    if density:
        sig_sum = float(sig_hist.values(flow=False).sum())
        if sig_sum > 0:
            scale = 1.0 / sig_sum
            sig_hist = sig_hist * scale
            ps_hist = ps_hist * scale

    sig_vals, edges, _ = _hist_arrays(sig_hist)
    ps_vals, _, ps_err = _hist_arrays(ps_hist)

    # Ranges
    xMin, xMax, yMin, yMax = _range_from_arrays(edges,
                                                [sig_vals],
                                                [ps_vals],
                                                logY=logY)

    # Ratio and errors
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(sig_vals > 0, ps_vals / sig_vals, 1.0)
        ps_rel = np.where(ps_vals > 0, ps_err / ps_vals, 0.0)
        ratio_err = ratio * ps_rel

    fig, (ax1, ax2) = subplots(2, 1, height_ratios=[4, 1])
    if logX: ax1.set_xscale('log')
    if logY: ax1.set_yscale('log')
    if logX: ax2.set_xscale('log')

    hep.histplot(sig_hist, histtype='step', lw=3,
                 label='Signal', color='tab:blue', ax=ax1)
    hep.histplot(ps_hist, yerr=ps_err, histtype='errorbar',
                 lw=2, color='black', marker='o', ms=4, linestyle='',
                 label='Pseudo-signal', ax=ax1)

    hep.histplot(ratio, bins=edges, yerr=ratio_err, histtype='band', ax=ax2)
    hep.histplot(ratio, bins=edges, histtype='errorbar', lw=2,
                 color='black', marker='o', ms=4, ax=ax2)
    ax2.axhline(1.0, color='gray', lw=2, alpha=0.6)
    ax2.axhline(pert, color='black', lw=2, ls='--', alpha=0.8)

    ax1.set_xlim(xMin, xMax)
    ax2.set_xlim(xMin, xMax)
    ax1.set_ylim(yMin, yMax)
    ax2.set_ylim(0.95, pert * 1.05)

    ax1.legend(loc='upper right', frameon=True)
    ax1.tick_params(labelbottom=False)

    lumi_text = rf'$\sqrt{{s}} = {ecm}$ GeV, {lumi} ab$^{{-1}}$'
    xtitle = r'm$_{\mathrm{recoil}}$ High [GeV]' if 'high' in sel else r'm$_{\mathrm{recoil}}$ Low [GeV]'
    ytitle = 'Normalized to Unity' if density else 'Events'
    set_labels(ax2, xtitle, 'Ratio', left='None')
    set_labels(ax1, '', ytitle, right=lumi_text)

    out = f'{outDir}/high' if 'high' in sel else f'{outDir}/low'
    mkdir(out)
    savefigs(fig, out, outName, suffix=f'_{target}', format=format)
    close(fig)
