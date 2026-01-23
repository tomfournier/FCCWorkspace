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

from .helper import (
    canvas_margins, 
    pad_margins, 
    configure_axis, 
    axis_limits, 
    y_offset, 
    setup_latex, 
    savecanvas
)
from ...tools.utils import mkdir



########################
### CONFIG AND SETUP ###
########################

# Disable interactive ROOT displays and remove default stat/title boxes
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

cfg = None  # Global configuration dictionary, populated at runtime



######################
### MAIN FUNCTIONS ###
######################

#________________________________
def canvas(width:  int = 1000, 
           height: int = 1000, 
           top:    float = 0.055,
           bottom: float = 0.11,
           left:   float = 0.15,
           right:  float = 0.05,
           batch: bool = False,
           yields: bool = False
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
    canvas_margins(c, 
                   top=top, 
                   bottom=bottom, 
                   left=left, 
                   right=right)

    # Apply log scales unless plotting yields
    if not yields:
        if cfg['logx']: c.SetLogx()
        if cfg['logy']: c.SetLogy()
    c.SetFillStyle(4000)
    if batch: c.SetTicks(1, 1)

    c.Modify()
    c.Update()

    return c

#_______________________________________
def canvasRatio(width:  int = 1000, 
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
    canvas_margins(c, 
                   top=0., 
                   bottom=0., 
                   left=0., 
                   right=0.)

    # Create upper pad for main plot and lower pad for ratio
    pad1 = ROOT.TPad('p1','p1', 0, cfg['ratiofraction'], 1, 1)
    pad2 = ROOT.TPad('p2','p2', 0, 0.0, 1, cfg['ratiofraction']-0.7*eps)

    pad_margins(pad1, top=0.055/(1.-cfg['ratiofraction']), bottom=eps,
                left=left)
    pad_margins(pad2, bottom=0.37, left=left)

    if cfg['logy']: pad1.SetLogy()
    if cfg['logx']:
        pad1.SetLogx()
        pad2.SetLogx()

    c.Modify()
    c.Update()

    return c, pad1, pad2

#_________________
def aux() -> None:
    '''
    Draw auxiliary text boxes with top-left and top-right labels on the plot.
    
    Uses global configuration 'topLeft' and 'topRight' strings to display
    plot metadata (e.g., luminosity, channel information).
    
    Returns:
        None
    '''
    y_off = y_offset(cfg['topRight'])
    
    # Draw left-aligned label at top-left
    latex = setup_latex(text_size=0.04, text_align=10)
    latex.DrawLatexNDC(0.15, 0.95, cfg['topLeft'])

    # Draw right-aligned label at top-right
    latex = setup_latex(text_size=0.04, text_align=30)
    latex.DrawLatex(0.95, y_off, cfg['topRight'])

#______________________
def auxRatio() -> None:
    '''
    Draw auxiliary text boxes for ratio plots with adaptive vertical positioning.
    
    Adjusts label positioning based on LaTeX special characters (superscripts,
    subscripts, square root symbols) in the 'topRight' configuration string.
    
    Returns:
        None
    '''
    # Detect special LaTeX formatting that affects vertical spacing
    has_sqrt = '#sqrt' in cfg['topRight']
    has_special = '^' in cfg['topRight'] or '_' in cfg['topRight']
    y_off = 0.935 if (has_sqrt and has_special) \
        else y_offset(cfg['topRight'], 0.945, 0.935)
    
    # Draw left-aligned label
    latex = setup_latex(text_size=0.06, text_align=13)
    latex.DrawLatex(0.15, 0.975, cfg['topLeft'])

    # Draw right-aligned label with computed offset
    latex = setup_latex(text_size=0.055, text_align=31)
    latex.DrawLatex(0.95, y_off, cfg['topRight'])

#________________________
def dummy(nbins: int = 1
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

    # Create empty histogram with specified bin count and range
    dummy = ROOT.TH1D('h', 'h', 
                      nbins, 
                      xmin, xmax)

    # Configure x-axis
    configure_axis(dummy.GetXaxis(), 
                   cfg['xtitle'], 
                   xmin, xmax,
                   title_offset=1.2, 
                   label_offset=1.2)
    # Configure y-axis
    configure_axis(dummy.GetYaxis(), 
                   cfg['ytitle'], 
                   ymin, ymax,
                   title_offset=1.7, 
                   label_offset=1.4)

    dummy.SetMinimum(ymin)
    dummy.SetMaximum(ymax)
    return dummy

#_________________________________________________________
def dummyRatio(nbins: int = 1, 
               rlines: list[float] = [1], 
               colors: list[ROOT.TColor] = [ROOT.kBlack]):
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
    xmin, xmax   = axis_limits(cfg, 'x')
    ymin, ymax   = axis_limits(cfg, 'y')
    yminR, ymaxR = axis_limits(cfg, 'y', ratio='R')

    # Create dummy histograms for upper (main) and lower (ratio) pads
    dummyT = ROOT.TH1D('h1', 'h', 
                       nbins, 
                       xmin, xmax)
    dummyB = ROOT.TH1D('h2', 'h', 
                       nbins, 
                       xmin, xmax)

    # Configure x-axis: hidden in upper pad, visible in lower pad
    configure_axis(dummyT.GetXaxis(), 
                   '', 
                   xmin, xmax,
                   title_size=0,
                   label_size=0, 
                   title_font=0,
                   label_font=0)
    configure_axis(dummyB.GetXaxis(), 
                   cfg['xtitle'], 
                   xmin, xmax,
                   title_size=32, 
                   label_size=28, 
                   title_offset=1.0,
                   label_offset=3.0)
    
    # Configure y-axes
    configure_axis(dummyT.GetYaxis(), 
                   cfg['ytitle'], 
                   ymin, ymax,
                   title_size=32, 
                   label_size=28, 
                   title_offset=1.7, 
                   label_offset=1.4)
    configure_axis(dummyB.GetYaxis(), 
                   cfg['ytitleR'], 
                   yminR, ymaxR, 
                   title_size=32, 
                   label_size=28, 
                   title_offset=1.7, 
                   label_offset=1.4)

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

#_________________________________________________________
def setup_cutflow_hist(n_cuts: int, 
                       labels_map: dict[str, str], 
                       cat: str
                       ) -> tuple[ROOT.TCanvas, ROOT.TH1]:
    '''
    Create and configure a histogram for cutflow plots.
    
    Sets up a canvas with grid and binned histogram, where each bin represents
    a selection cut with corresponding label.
    
    Args:
        n_cuts (int): Number of cuts (bins) in the histogram.
        labels_map (dict[str, str]): Mapping of cut keys to display labels.
        cat (str): Category string (e.g., 'ee', 'mumu') to substitute in labels.
    
    Returns:
        tuple: (canvas, configured_histogram).
    '''
    c = canvas()
    c.SetGrid()
    c.SetTicks()
    d = dummy(n_cuts)

    # Adjust label size and offset for readability
    d.GetXaxis().SetLabelSize(0.75 * d.GetXaxis().GetLabelSize())
    d.GetXaxis().SetLabelOffset(1.3 * d.GetXaxis().GetLabelOffset())
    # Set bin labels with category substitution
    for i, cut in enumerate(labels_map):
        d.GetXaxis().SetBinLabel(i+1, labels_map[cut].replace('#ell', cat))
    # Rotate labels for better visibility
    d.GetXaxis().LabelsOption('u')

    return c, d

#________________________________________
def finalize_canvas(canvas: ROOT.TCanvas,
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
    if grid:
        canvas.SetGrid()
    canvas.Modify()
    canvas.Update()
    aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

#_____________________________________________
def save_canvas(canvas: ROOT.TCanvas, 
                outDir: str, 
                outName: str, 
                suffix: str = '', 
                plot_file: list[str] = ['png']
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
    mkdir(outDir)

    # Apply final formatting before saving
    aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    savecanvas(canvas, outDir, outName, suffix, plot_file)
