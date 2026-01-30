'''ROOT histogram and axis configuration helpers for physics analysis plots.

Provides:
- Configuration management: `make_cfg()`, `build_cfg()`.
- Canvas and pad layout: `canvas_margins()`, `pad_margins()`.
- Legend creation: `mk_legend()`.
- Axis formatting: `configure_axis()`, `axis_limits()`.
- Histogram styling: `style_hist()`.
- Text annotation: `setup_latex()`, `draw_latex()`, `y_offset()`.
- Histogram loading with caching: `load_hists()`, `_get_hist_cached()`.
- File I/O: `savecanvas()`, `save_plot()`.

Functions:
- `make_cfg()`: Complete plotting configuration with defaults and validation.
- `build_cfg()`: Build full configuration from histogram and axis parameters.
- `canvas_margins()`: Set canvas margins with optional values.
- `pad_margins()`: Configure margins for ROOT pads.
- `mk_legend()`: Create configured legend with automatic sizing based on entry count.
- `load_hists()`: Load histograms for multiple processes with LRU caching.
- `_get_hist_cached()`: Internal cached histogram loader (LRU cached).
- `axis_limits()`: Extract and apply log-scale padding to axis ranges.
- `configure_axis()`: Set axis title, range, fonts, and offsets in one call.
- `style_hist()`: Apply line/fill color, width, style, and scaling to histogram.
- `setup_latex()`: Create TLatex object with NDC mode and styling.
- `y_offset()`: Compute adaptive vertical offset for super/subscript text.
- `draw_latex()`: Draw multiple text annotations with individual sizing.
- `savecanvas()`: Export canvas to multiple file formats.
- `save_plot()`: Save canvas with automatic directory creation.

Conventions:
- Configuration dictionaries contain keys: xmin, xmax, ymin, ymax, logx, logy, xtitle, ytitle, topLeft, topRight.
- Ratio plot configurations use suffixed keys: yminR, ymaxR, ytitleR, ratiofraction.
- All axis sizes specified in absolute points (font code 43) unless otherwise noted.
- Margins specified as fractions of canvas/pad dimensions (0-1 range).
- Log-scale ranges padded by ±0.1% to prevent edge clipping in zoomed plots.
- Histogram caching via LRU (128-entry cache) reduces repeated file I/O for common variables.
- Text positioning uses NDC (normalized device coordinates) for frame-independent placement.

Usage:
- Build complete plot configurations with automatic axis range calculation and log-scale handling.
- Configure ROOT axes with consistent fonts, sizes, and offsets across multiple canvases.
- Style multiple histograms efficiently using helper functions for colors and line properties.
- Manage legends automatically sized based on entry count with configurable layout.
- Export plots to disk with support for multiple formats and optional directory creation.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import os, ROOT

from functools import lru_cache
from typing import Union

from ...config import warning
from ...tools.utils import mkdir
from ...tools.process import getHist, get_range



######################
### MAIN FUNCITONS ###
######################

#___________________________________________
def make_cfg(
    cfg: dict[str, 
              Union[str, float, int, bool]], 
    ecm: int = 240, 
    lumi: float = 10.8,
    ratio_plot: bool = False
    ) -> dict[str, 
              Union[str, float, int, None]]:
    '''Complete plotting configuration with defaults and validation.
    
    Args:
        cfg (dict[str, str | float | int | bool]): Partial configuration dictionary with plot settings.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        ratio_plot (bool, optional): Whether ratio plot is enabled. Defaults to False.
    
    Returns:
        dict[str, str | float | int | None]: Complete configuration dictionary with all required fields.
    '''

    # Validate required x-y range parameters
    if ('xmin' not in cfg) or ('xmax' not in cfg) \
        or ('ymin' not in cfg) or ('ymax' not in cfg):
        msg = 'Histogram limits not set. Aborting code'
        warning(msg)

    # Set default x-y scale options
    cfg.setdefault('logx', False)
    cfg.setdefault('logy', False)
    
    # Set default title labels
    cfg.setdefault('xtitle', '')
    cfg.setdefault('ytitle', 'Events')
    cfg.setdefault('topLeft', '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}')
    cfg.setdefault('topRight', f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{-1}}')

    # Configure ratio plot settings if enabled
    if (('ymin' not in cfg) or ('ymax' not in cfg)) and ratio_plot:
        msg = 'Ratio limits of the histogram not set. Aborting code'
        warning(msg)
    cfg.setdefault('ytitleR', 'Ratio')
    cfg.setdefault('ratiofraction', 0.3)
    
    return cfg

#____________________________________
def build_cfg(
    hist: ROOT.TH1, 
    logX: bool = False, 
    logY: bool = False,
    xmin: Union[float, None] = None,
    xmax: Union[float, None] = None,
    ymin: Union[float, None] = None, 
    ymax: Union[float, None] = None,
    xtitle: str = '',
    ytitle: str = 'Events',
    ecm: int = 240, 
    lumi: float = 10.8,
    strict: bool = True, 
    stack: bool = False,
    hists: Union[list, None] = None,
    range_func: callable = get_range,
    cutflow: bool = False,
    decay: bool = False
    ) -> dict:
    '''Build complete plotting configuration with computed axis ranges.
    
    Args:
        hist (ROOT.TH1): Reference histogram for range calculation.
        logX (bool, optional): Enable logarithmic x-axis. Defaults to False.
        logY (bool, optional): Enable logarithmic y-axis. Defaults to False.
        xmin (float | None, optional): Minimum x-axis value (auto-computed if None). Defaults to None.
        xmax (float | None, optional): Maximum x-axis value (auto-computed if None). Defaults to None.
        ymin (float | None, optional): Minimum y-axis value (auto-computed if None). Defaults to None.
        ymax (float | None, optional): Maximum y-axis value (auto-computed if None). Defaults to None.
        xtitle (str, optional): X-axis label (uses histogram title if empty). Defaults to ''.
        ytitle (str, optional): Y-axis label. Defaults to 'Events'.
        ecm (int, optional): Center-of-mass energy in GeV. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        strict (bool, optional): Use strict range calculation. Defaults to True.
        stack (bool, optional): Whether histograms are stacked. Defaults to False.
        hists (list | None, optional): Additional histograms for range calculation. Defaults to None.
        range_func (callable, optional): Function to compute axis ranges. Defaults to get_range.
        cutflow (bool, optional): Use cutflow-specific range handling. Defaults to False.
        decay (bool, optional): Use decay-specific range handling. Defaults to False.
    
    Returns:
        dict: Complete plotting configuration dictionary.
    '''
    # Adjust scale factors for log/linear y-axis
    scale_min, scale_max = 5e-1 if logY else 1, 1e4 if logY else 1.5
    if not cutflow:
        if not decay:
            xMin, xMax, yMin, yMax = range_func(
                [hist], hists, logY=logY, stack=stack, strict=strict,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                scale_min=scale_min, scale_max=scale_max
            )
        else:
            xMin, xMax, yMin, yMax = range_func(
                hists, logY=logY,  strict=strict,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                scale_min=scale_min, scale_max=scale_max
            )
    else:
        if (xmin is None) or (xmax is None) or \
            (ymin is None) or (ymax is None):
            warning('Range was not set, aborting...')
        xMin, xMax, yMin, yMax = xmin, xmax, ymin, ymax
    
    # Determine x-axis title from parameter or histogram
    if xtitle=='':
        xTitle = hist.GetXaxis().GetTitle()
    elif xtitle=='None':
        xTitle = ''
    else: 
        xTitle = xtitle
    return make_cfg({
        'xmin': xMin, 'xmax': xMax, 
        'ymin': yMin, 'ymax': yMax,
        'logx': logX, 'logy': logY,
        'xtitle': xTitle, 
        'ytitle': ytitle,
    }, ecm=ecm, lumi=lumi)

#______________________________________
def canvas_margins(
    c: ROOT.TCanvas, 
    top:    Union[float, None] = 0.055, 
    bottom: Union[float, None] = 0.11,
    left:   Union[float, None] = 0.15, 
    right:  Union[float, None] = 0.05
    ) -> None:
    '''Set canvas margins with optional values.
    
    Args:
        c (ROOT.TCanvas): ROOT canvas to configure.
        top (float | None, optional): Top margin (skipped if None). Defaults to 0.055.
        bottom (float | None, optional): Bottom margin (skipped if None). Defaults to 0.11.
        left (float | None, optional): Left margin (skipped if None). Defaults to 0.15.
        right (float | None, optional): Right margin (skipped if None). Defaults to 0.05.
    
    Returns:
        None
    '''
    if top is not None:
        c.SetTopMargin(top)
    if bottom is not None:
        c.SetBottomMargin(bottom)
    if left is not None:
        c.SetLeftMargin(left)
    if right is not None:
        c.SetRightMargin(right)

#________________________
def pad_margins(
    pad: ROOT.TPad, 
    top:    float = 0.0, 
    bottom: float = 0.0,
    left:   float = 0.15,
    right:  float = 0.05
    ) -> None:
    '''Set margins for a ROOT pad.
    
    Args:
        pad (ROOT.TPad): ROOT pad to configure.
        top (float, optional): Top margin. Defaults to 0.0.
        bottom (float, optional): Bottom margin. Defaults to 0.0.
        left (float, optional): Left margin. Defaults to 0.15.
        right (float, optional): Right margin. Defaults to 0.05.
    
    Returns:
        None
    '''
    pad.SetTopMargin(top)
    pad.SetBottomMargin(bottom)
    pad.SetLeftMargin(left)
    pad.SetRightMargin(right)

#____________________________
def mk_legend(
    num_entries: int, 
    columns: int = 1, 
    x1: float = 0.55, 
    y1: float = 0.99, 
    x2: float = 0.99, 
    y2: float = 0.90,
    border_size: int = 0,
    fill_style:  int = 0,
    text_size:  float = 0.03,
    set_margin: float = 0.2,
    text_font: int = -1
    ) -> ROOT.TLegend:
    '''Create configured ROOT legend with automatic sizing.
    
    Args:
        num_entries (int): Number of legend entries.
        columns (int, optional): Number of legend columns. Defaults to 1.
        x1 (float, optional): Left edge in NDC coordinates. Defaults to 0.55.
        y1 (float, optional): Top edge in NDC coordinates (adjusted by num_entries). Defaults to 0.99.
        x2 (float, optional): Right edge in NDC coordinates. Defaults to 0.99.
        y2 (float, optional): Fixed top position in NDC coordinates. Defaults to 0.90.
        border_size (int, optional): Legend border width. Defaults to 0.
        fill_style (int, optional): Legend fill style. Defaults to 0.
        text_size (float, optional): Legend text size. Defaults to 0.03.
        set_margin (float, optional): Legend entry margin. Defaults to 0.2.
        text_font (int, optional): Legend text font (-1 uses default). Defaults to -1.
    
    Returns:
        ROOT.TLegend: Configured ROOT TLegend object.
    '''
    leg = ROOT.TLegend(x1, y1 - (num_entries) \
                       * 0.06 * (1/columns), 
                       x2, y2)
    if text_font!=-1:
        leg.SetTextFont(text_font)
    leg.SetBorderSize(border_size)
    leg.SetFillStyle(fill_style)
    leg.SetTextSize(text_size)
    leg.SetMargin(set_margin)
    leg.SetNColumns(columns)
    return leg

@lru_cache(maxsize=128)
def _get_hist_cached(
    hName: str,
    procs: tuple,
    inDir: str,
    suffix: str,
    rebin: int,
    lazy: bool
    ) -> ROOT.TH1:
    '''Load histogram with LRU caching to reduce file I/O.
    
    Args:
        hName (str): Histogram name/variable.
        procs (tuple): Process names (tuple for hashability).
        inDir (str): Input directory path.
        suffix (str): File suffix.
        rebin (int): Rebinning factor.
        lazy (bool): Enable lazy loading.
    
    Returns:
        ROOT.TH1: Loaded and optionally rebinned histogram.
    '''
    return getHist(
        hName, list(procs), inDir, 
        suffix=suffix, rebin=rebin, lazy=lazy
        )

#______________________________
def load_hists(
    processes: dict[str, 
                    list[str]], 
    variable: str, 
    inDir: str, 
    suffix: str, 
    rebin: int = 1, 
    lazy: bool = True
    ) -> dict[str, ROOT.TH1]:
    '''Load histograms for all specified processes.
    
    Args:
        processes (dict[str, list[str]]): Dictionary mapping process names to process lists.
        variable (str): Histogram variable name.
        inDir (str): Input directory path.
        suffix (str): File suffix.
        rebin (int, optional): Rebinning factor. Defaults to 1.
        lazy (bool, optional): Enable lazy loading. Defaults to True.
    
    Returns:
        dict[str, ROOT.TH1]: Dictionary mapping process names to histograms.
    '''
    return {proc: _get_hist_cached(
                    variable, tuple(proc_list), 
                    inDir, suffix=suffix, 
                    rebin=rebin, lazy=lazy
                )
            for proc, proc_list in processes.items()}

#___________________________________________
def axis_limits(
    cfg: dict[str, 
              Union[str, float, int, bool]], 
    axis: str, 
    ratio: str = ''
    ) -> tuple[float, float]:
    '''Extract axis range from configuration with log scale padding.
    
    Args:
        cfg (dict[str, str | float | int | bool]): Plotting configuration dictionary.
        axis (str): Axis name ('x' or 'y').
        ratio (str, optional): Suffix for ratio plot axes (e.g., 'R'). Defaults to ''.
    
    Returns:
        tuple: (min, max) axis limits with optional log padding.
    '''
    is_log = cfg[f'log{axis}']
    min = float(cfg[f'{axis}min{ratio}'])
    max = float(cfg[f'{axis}max{ratio}'])
    
    # Apply small padding for log scale to prevent edge clipping
    if is_log:
        return 0.999 * min, 1.001 * max
    return min, max

#_____________________________
def configure_axis(
    axis, 
    title: str, 
    axis_min:     float, 
    axis_max:     float,
    title_size:   int = 40, 
    label_size:   int = 35, 
    title_offset: float = 1.2, 
    label_offset: float = 1.2, 
    title_font:   int = 43,
    label_font:   int = 43
    ) -> None:
    '''Configure axis styling, range, and typography.
    
    Args:
        axis (ROOT.TAxis): ROOT axis object (TAxis).
        title (str): Axis title text.
        axis_min (float): Minimum axis value.
        axis_max (float): Maximum axis value.
        title_size (int, optional): Title font size. Defaults to 40.
        label_size (int, optional): Label font size. Defaults to 35.
        title_offset (float, optional): Title offset multiplier. Defaults to 1.2.
        label_offset (float, optional): Label offset multiplier. Defaults to 1.2.
        title_font (int, optional): Title font code. Defaults to 43.
        label_font (int, optional): Label font code. Defaults to 43.
    
    Returns:
        None
    '''
    if title:
        axis.SetTitle(title)
    axis.SetRangeUser(axis_min, axis_max)
    axis.SetTitleSize(title_size)
    axis.SetLabelSize(label_size)
    axis.SetTitleFont(title_font)
    axis.SetLabelFont(label_font)
    axis.SetTitleOffset(title_offset * axis.GetTitleOffset())
    axis.SetLabelOffset(label_offset * axis.GetLabelOffset())

#______________________________________
def style_hist(
    hist: ROOT.TH1, 
    color: int, 
    width: int = 1, 
    style: int = 1, 
    scale: float = 1., 
    fill_color: Union[int, None] = None
    ) -> None:
    '''Apply visual styling and optional scaling to histogram.
    
    Args:
        hist (ROOT.TH1): ROOT histogram to style.
        color (int): Line color code.
        width (int, optional): Line width. Defaults to 1.
        style (int, optional): Line style code. Defaults to 1.
        scale (float, optional): Scaling factor for histogram values. Defaults to 1.
        fill_color (int | None, optional): Fill color code (None disables fill). Defaults to None.
    
    Returns:
        None
    '''
    hist.SetLineColor(color)
    hist.SetLineWidth(width)
    hist.SetLineStyle(style)
    if fill_color is not None:
        hist.SetFillColor(fill_color)
    if scale != 1.:
        hist.Scale(scale)

#______________________________________________
def style_hists_batch(
    hists: list[ROOT.TH1],
    colors: list[int],
    widths: list[int] = None,
    scales: list[float] = None,
    fill_colors: list[Union[int, None]] = None
    ) -> None:
    '''Apply styling to multiple histograms in batch for performance.
    
    Optimized version for styling many histograms at once, reducing Python
    call overhead compared to looping with style_hist().
    
    Args:
        hists (list[ROOT.TH1]): List of ROOT histograms to style.
        colors (list[int]): Line colors for each histogram.
        widths (list[int], optional): Line widths (default 1 for all). Defaults to None.
        scales (list[float], optional): Scale factors (default 1 for all). Defaults to None.
        fill_colors (list[int | None], optional): Fill colors (default None). Defaults to None.
    
    Returns:
        None
    '''
    n = len(hists)
    widths = widths or [1] * n
    scales = scales or [1.] * n
    fill_colors = fill_colors or [None] * n
    
    for i, hist in enumerate(hists):
        hist.SetLineColor(colors[i])
        hist.SetLineWidth(widths[i])
        if fill_colors[i] is not None:
            hist.SetFillColor(fill_colors[i])
        if scales[i] != 1.:
            hist.Scale(scales[i])

#___________________________________________
def setup_latex(
    text_size: float, 
    text_align: int, 
    text_color: Union[int, ROOT.TColor] = 1,
    text_font: int = 42
    ) -> ROOT.TLatex:
    '''Create TLatex object for text annotations.
    
    Args:
        text_size (float): Text size in NDC coordinates.
        text_align (int): Text alignment code.
        text_color (int | ROOT.TColor, optional): Text color code or TColor object. Defaults to 1.
        text_font (int, optional): Text font code. Defaults to 42.
    
    Returns:
        ROOT.TLatex: Configured TLatex object with NDC enabled.
    '''
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(text_size)
    latex.SetTextColor(text_color)
    latex.SetTextFont(text_font)
    latex.SetTextAlign(text_align)
    return latex

#_______________________
def y_offset(
    text: str, 
    high: float = 0.955, 
    low:  float = 0.945
    ) -> float:
    '''Compute vertical offset to prevent superscript/subscript clipping.
    
    Args:
        text (str): Text string to check for LaTeX markup.
        high (float, optional): Default y-position for plain text. Defaults to 0.955.
        low (float, optional): Adjusted y-position for text with super/subscripts. Defaults to 0.945.
    
    Returns:
        float: Y-coordinate in NDC units.
    '''
    has_underscore = '_' in text
    has_caret = '^' in text
    return low if (has_underscore or has_caret) else high

#_______________________________________
def draw_latex(
    latex: ROOT.TLatex, 
    text_data: list[tuple[str, float, 
                          float, float]]
    ) -> None:
    '''Draw multiple text annotations with individual sizing.
    
    Args:
        latex (ROOT.TLatex): Configured TLatex object.
        text_data (list[tuple[str, float, float, float]]): List of tuples (text, x, y, size) for each annotation.
    
    Returns:
        None
    '''
    for text, x, y, size in text_data:
        latex.SetTextSize(size)
        latex.DrawLatex(x, y, text)

#______________________________
def savecanvas(
    c: ROOT.TCanvas, 
    outDir: str, 
    plotname: str,
    suffix: str = '', 
    format: list[str] = ['png']
    ) -> None:
    '''Export canvas to multiple file formats.
    
    Args:
        c (ROOT.TCanvas): ROOT canvas to save.
        outDir (str): Output directory path.
        plotname (str): Base filename without extension.
        suffix (str, optional): Optional filename suffix. Defaults to ''.
        format (list[str], optional): List of file formats (e.g., ['png', 'pdf']). Defaults to ['png'].
    
    Returns:
        None
    '''
    fpath = os.path.join(outDir, plotname+suffix)
    for f in format:
        c.SaveAs(f'{fpath}.{f}')

#________________________
def save_plot(
    canvas: ROOT.TCanvas, 
    outDir: str, 
    outName: str, 
    suffix: str,
    format: list[str], 
    ) -> None:
    '''Save canvas with automatic directory creation.
    
    Args:
        canvas (ROOT.TCanvas): ROOT canvas to save.
        outDir (str): Output directory path (created if missing).
        outName (str): Base filename without extension.
        suffix (str): Optional filename suffix.
        format (list[str]): List of file formats (e.g., ['png', 'pdf']).
    
    Returns:
        None
    '''
    mkdir(outDir)
    savecanvas(
        canvas, outDir, outName, 
        suffix=suffix, format=format
    )
