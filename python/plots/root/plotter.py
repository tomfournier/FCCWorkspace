'''ROOT canvas and histogram utilities for physics analysis plots.

Provides:
- Canvas creation and configuration: `canvas()`, `canvasRatio()`, `setup_cutflow_hist()`.
- Dummy histograms as plot templates: `dummy()`, `dummyRatio()`.
- Auxiliary label rendering: `aux()`, `auxRatio()`.
- Canvas finalization and file export: `finalize_canvas()`, `save_canvas()`.
- Global configuration management via module-level `cfg` dictionary.
- Integration with ROOT styling and axis formatting helpers.

Functions:
- `canvas()`: Create standard ROOT canvas with configured margins and log scales.
- `canvasRatio()`: Create two-pad canvas with ratio plot layout and spacing.
- `dummy()`: Generate template histogram with axis labels and configured limits.
- `dummyRatio()`: Generate dual dummy histograms with reference lines for ratio plots.
- `aux()`: Render top-left and top-right labels with metadata (luminosity, channel).
- `auxRatio()`: Render labels for ratio plots with adaptive vertical positioning.
- `setup_cutflow_hist()`: Configure canvas and histogram for cutflow visualization.
- `finalize_canvas()`: Apply final cosmetics (grid, axis redraw, auxiliary labels).
- `save_canvas()`: Save canvas to file in multiple formats with proper formatting.

Conventions:
- Global `cfg` dictionary populated at runtime with plot configuration.
- All canvases created in batch mode (ROOT.gROOT.SetBatch(True)).
- Stat and title boxes disabled by default for cleaner appearance.
- Margins and axis labels configurable per canvas type (standard, ratio, cutflow).
- Logarithmic scaling on both axes controlled via `cfg['logx']` and `cfg['logy']`.
- Reference lines in ratio plots colored and styled via helper functions.
- Output directories created automatically; multiple formats supported (png, pdf, etc.).

Usage:
- Create publication-quality ROOT plots with standard FCC-ee styling conventions.
- Build ratio plots with dual pads for data/MC comparison or signal/background ratios.
- Generate cutflow histograms with bin-per-cut and automatic label substitution.
- Export finished plots to disk with automatic path creation and format conversion.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import ROOT

from tools.utils import mkdir



########################
### CONFIG AND SETUP ###
########################

# Disable interactive ROOT displays and remove default stat/title boxes
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

# Pre-warm ROOT graphics system to reduce first-plot latency
# This forces initialization of graphics drivers, fonts, and I/O systems
_warmup_canvas = ROOT.TCanvas('_warmup', '_warmup', 100, 100)
_warmup_canvas.Close()
del _warmup_canvas

cfg = {}  # Global configuration dictionary, populated at runtime



########################
### HELPER FUNCTIONS ###
########################

def canvas_margins(
    c: ROOT.TCanvas,
    top:    float | None = 0.055,
    bottom: float | None = 0.11,
    left:   float | None = 0.15,
    right:  float | None = 0.05
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
    return None


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
    
    return None


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
    has_caret      = '^' in text
    return low if (has_underscore or has_caret) else high


def setup_latex(
    text_size: float,
    text_align: int,
    text_color: int | ROOT.TColor = 1,
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


def axis_limits(
    cfg: dict[str, str | float | int | bool],
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
    try:
        min = float(cfg[f'{axis}min{ratio}'])
        max = float(cfg[f'{axis}max{ratio}'])
    except KeyError:
        raise KeyError('Limit should be defined in cfg')

    # Apply small padding for log scale to prevent edge clipping
    if cfg.get(f'log{axis}'): return 0.999 * min, 1.001 * max
    return min, max


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
    if title: axis.SetTitle(title)
    axis.SetRangeUser(axis_min, axis_max)
    axis.SetTitleSize(title_size)
    axis.SetLabelSize(label_size)
    axis.SetTitleFont(title_font)
    axis.SetLabelFont(label_font)
    axis.SetTitleOffset(title_offset * axis.GetTitleOffset())
    axis.SetLabelOffset(label_offset * axis.GetLabelOffset())

    return None



######################
### MAIN FUNCTIONS ###
######################

def canvas(
    width:  int = 1000,
    height: int = 1000,
    top:    float = 0.055,
    bottom: float = 0.11,
    left:   float = 0.15,
    right:  float = 0.05,
    set_ticks: bool = False,
     ) -> ROOT.TCanvas:
    '''
    Create a configured ROOT canvas with standard margins and axis settings.

    Args:
        width (int, optional): Canvas width in pixels. Defaults to 1000.
        height (int, optional): Canvas height in pixels. Defaults to 1000.
        top (float, optional): Top margin fraction. Defaults to 0.055.
        bottom (float, optional): Bottom margin fraction. Defaults to 0.11.
        left (float, optional): Left margin fraction. Defaults to 0.15.
        right (float, optional): Right margin fraction. Defaults to 0.05.
        batch (bool, optional): If True, enable tick marks on all sides. Defaults to False.
        yields (bool, optional): If True, disable log-scale settings. Defaults to False.

    Returns:
        ROOT.TCanvas: Configured ROOT.TCanvas object.
    '''
    c = ROOT.TCanvas('c', 'c', width, height)
    canvas_margins(c, top, bottom, left, right)

    # Apply log scales unless plotting yields
    if cfg.get('logx'): c.SetLogx()
    if cfg.get('logy'): c.SetLogy()
    c.SetFillStyle(4000)
    if set_ticks: c.SetTicks(1, 1)

    c.Modify()
    c.Update()

    return c


def canvasRatio(
    width:  int = 1000,
    height: int = 1000,
    left: float = 0.15,
    eps:  float = 0.025
     ) -> tuple[ROOT.TCanvas,
                ROOT.TPad,
                ROOT.TPad]:
    '''
    Create a canvas with two vertically-stacked pads for ratio plots.

    Args:
        width (int, optional): Canvas width in pixels. Defaults to 1000.
        height (int, optional): Canvas height in pixels. Defaults to 1000.
        left (float, optional): Left margin fraction. Defaults to 0.15.
        eps (float, optional): Spacing between pads as fraction of canvas height. Defaults to 0.025.

    Returns:
        tuple: (canvas, upper_pad, lower_pad) configured for ratio plots.
    '''
    c = ROOT.TCanvas('c', 'c', width, height)
    canvas_margins(c, 0, 0, 0, 0)

    rfrac = cfg.get('ratiofraction', '')

    # Create upper pad for main plot and lower pad for ratio
    pad1 = ROOT.TPad('p1','p1', 0, rfrac, 1, 1)
    pad2 = ROOT.TPad('p2','p2', 0, 0, 1, rfrac - 0.7*eps)

    pad_margins(pad1, 0.055/(1 - rfrac), eps, left)
    pad_margins(pad2, 0, 0.37, left)

    if cfg.get('logx'): 
        pad1.SetLogx()
        pad2.SetLogx()
    if cfg.get('logy'):
        pad1.SetLogy()

    c.Modify()
    c.Update()

    return c, pad1, pad2


def aux() -> None:
    '''
    Draw auxiliary text boxes with top-left and top-right labels on the plot.

    Uses global configuration 'topLeft' and 'topRight' strings to display
    plot metadata (e.g., luminosity, channel information).

    Returns:
        None
    '''
    y_off = y_offset(cfg.get('topRight', ''))

    # Draw left-aligned label at top-left
    latex = setup_latex(0.04, 10)
    latex.DrawLatexNDC(0.15, 0.95, cfg.get('topLeft', ''))

    # Draw right-aligned label at top-right
    latex = setup_latex(0.04, 30)
    latex.DrawLatex(0.95, y_off, cfg.get('topRight', ''))

    return None


def auxRatio() -> None:
    '''
    Draw auxiliary text boxes for ratio plots with adaptive vertical positioning.

    Adjusts label positioning based on LaTeX special characters (superscripts,
    subscripts, square root symbols) in the 'topRight' configuration string.

    Returns:
        None
    '''

    topleft  = cfg.get('topLeft',  '')
    topright = cfg.get('topRight', '')

    # Detect special LaTeX formatting that affects vertical spacing
    has_sqrt = '#sqrt' in topright
    has_special = '^' in topright or '_' in topright
    y_off = 0.935 if (has_sqrt and has_special) \
        else y_offset(topright, 0.945, 0.935)

    # Draw left-aligned label
    latex = setup_latex(0.06, 13)
    latex.DrawLatex(0.15, 0.975, topleft)

    # Draw right-aligned label with computed offset
    latex = setup_latex(0.055, 31)
    latex.DrawLatex(0.95, y_off, topright)

    return None


def dummy(
        nbins: int = 1,
        labels: list[str] = [],
        label_size: float = 1,
        label_offset: float = 1.2,
) -> ROOT.TH1D:
    '''
    Create a dummy histogram with configured axis limits and labels.

    The dummy histogram serves as a template for plot appearance without
    containing actual data. Useful for setting axis ranges and titles.

    Args:
        nbins (int, optional): Number of histogram bins. Defaults to 1.

    Returns:
        ROOT.TH1D: Configured ROOT.TH1D histogram with axis labels and limits set.
    '''
    xmin, xmax = axis_limits(cfg, 'x')
    ymin, ymax = axis_limits(cfg, 'y')

    xtitle = cfg.get('xtitle', '')
    ytitle = cfg.get('ytitle', '')

    # Create empty histogram with specified bin count and range
    dummy = ROOT.TH1D('h', 'h', nbins, xmin, xmax)

    # Configure x-axis
    configure_axis(dummy.GetXaxis(), xtitle,
                   xmin, xmax,
                   label_size=label_size, title_offset=1.2,
                   label_offset=label_offset)
    # Configure y-axis
    configure_axis(dummy.GetYaxis(), ytitle, ymin,
                   ymax, title_offset=1.7, label_offset=1.4)

    dummy.SetMinimum(ymin)
    dummy.SetMaximum(ymax)

    if nbins > 1:
        if len(labels) != nbins:
            raise ValueError('labels should have the same size as nbins')
        for i, label in enumerate(labels):
            dummy.GetXaxis().SetBinLabel(i+1, label)
            dummy.GetXaxis().LabelsOption('u')

    return dummy


def dummyRatio(
    nbins: int = 1,
    rlines: list[float] = [1],
    colors: list[ROOT.TColor] = [ROOT.kBlack]
     ) -> tuple[ROOT.TH1D, ROOT.TH1D, list[ROOT.TLine]]:
    '''
    Create dummy histograms for ratio plots with reference lines.

    Generates two stacked dummy histograms (main and ratio) with configured
    axes, and optional reference lines for ratio comparison.

    Args:
        nbins (int, optional): Number of histogram bins. Defaults to 1.
        rlines (list[float], optional): Y-values for horizontal reference lines in ratio pad. Defaults to [1].
        colors (list[ROOT.TColor], optional): Colors for reference lines (one per line). Defaults to [ROOT.kBlack].

    Returns:
        tuple: (upper_dummy, lower_dummy, line_objects) for ratio plots.
    '''
    xmin,  xmax  = axis_limits(cfg, 'x')
    ymin,  ymax  = axis_limits(cfg, 'y')
    yminR, ymaxR = axis_limits(cfg, 'y', ratio='R')

    xtitle  = cfg.get('xtitle',  '')
    ytitle  = cfg.get('ytitle',  '')
    ytitleR = cfg.get('ytitleR', '')

    # Create dummy histograms for upper (main) and lower (ratio) pads
    dummyT = ROOT.TH1D('h1', 'h', nbins, xmin, xmax)
    dummyB = ROOT.TH1D('h2', 'h', nbins, xmin, xmax)

    # Configure x-axis: hidden in upper pad, visible in lower pad
    configure_axis(dummyT.GetXaxis(), '', xmin, xmax,
                   0, 0, 0, 0)
    configure_axis(dummyB.GetXaxis(), xtitle,
                   xmin, xmax,
                   32, 28,
                   1.0, 3.0)

    # Configure y-axes
    configure_axis(dummyT.GetYaxis(), ytitle,
                   ymin, ymax,
                   32, 28,
                   1.7, 1.4)
    configure_axis(dummyB.GetYaxis(), ytitleR,
                   yminR, ymaxR,
                   32, 28,
                   1.7, 1.4)

    dummyT.SetMaximum(ymax)
    dummyT.SetMinimum(ymin)
    dummyB.SetMinimum(yminR)
    dummyB.SetMaximum(ymaxR)
    dummyB.GetYaxis().SetNdivisions(505)

    # Create reference lines at specified y-values
    lines = []
    for rline, color in zip(rlines, colors):
        line = ROOT.TLine(xmin, rline, xmax, rline)
        line.SetLineColor(color), line.SetLineWidth(2)
        lines.append(line)

    return dummyT, dummyB, lines


def finalize_canvas(
    canvas: ROOT.TCanvas,
    grid: bool = True
     ) -> None:
    '''
    Finalize canvas appearance and redraw elements.

    Applies grid, ticks, auxiliary labels, and refreshes the canvas display.

    Args:
        canvas (ROOT.TCanvas): ROOT.TCanvas to finalize.
        grid (bool, optional): If True, enable grid lines on the canvas. Defaults to True.

    Returns:
        None
    '''
    if grid: canvas.SetGrid()
    canvas.Modify()
    canvas.Update()
    aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()


def save_canvas(
    canvas: ROOT.TCanvas,
    outDir: str,
    outName: str,
    suffix: str = '',
    file_formats: list[str] = ['png'],
    quiet: bool = False
) -> None:
    '''
    Save canvas to file with auxiliary labels and proper formatting.

    Creates output directory if needed, applies final cosmetics (axis redraw,
    auxiliary labels), and exports to specified file formats.

    Args:
        canvas (ROOT.TCanvas): ROOT.TCanvas to save.
        outDir (str): Output directory path.
        outName (str): Base filename for output (without extension).
        suffix (str, optional): Optional suffix to append to filename before extension. Defaults to ''.
        plot_file (list[str], optional): List of file formats to save (e.g., ['png', 'pdf']). Defaults to ['png'].

    Returns:
        None
    '''
    import os
    mkdir(outDir)

    # Apply final formatting before saving
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()

    fpath = os.path.join(outDir, outName+suffix)
    previous_level = ROOT.gErrorIgnoreLevel
    try:
        if quiet:
            ROOT.gErrorIgnoreLevel = ROOT.kWarning
        for f in file_formats:
            canvas.SaveAs(f'{fpath}.{f}')
    finally:
        ROOT.gErrorIgnoreLevel = previous_level
