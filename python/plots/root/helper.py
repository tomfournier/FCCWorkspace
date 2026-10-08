'''ROOT histogram and axis configuration helpers for physics analysis plots.

Provides:
- Configuration management: `make_cfg()`, `build_cfg()`.
- Canvas and pad layout: `canvas_margins()`, `pad_margins()`.
- Legend creation: `mk_legend()`.
- Axis formatting: `configure_axis()`, `axis_limits()`.
- Histogram styling: `style_hist()`, `style_hists_batch()`.
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
- `style_hists_batch()`: Apply styling to multiple histograms in batch for performance.
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
- Log-scale ranges padded by ±0.1% (0.999x–1.001x) to prevent edge clipping in zoomed plots.
- Histogram caching via LRU (128-entry cache) reduces repeated file I/O for common variables.
- Text positioning uses NDC (normalized device coordinates) for frame-independent placement.
- Batch styling operations via `style_hists_batch()` minimize Python call overhead for multiple histograms.

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

import ROOT

from typing import Any, Union

from logger import get_logger
LOGGER = get_logger(__name__)


######################
### MAIN FUNCTIONS ###
######################

def make_cfg(
    cfg: dict[str, str | float | int | bool],
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
        LOGGER.error('Histogram limits not set. Aborting code')
        exit(1)

    # Set default x-y scale options
    cfg.setdefault('logx', False)
    cfg.setdefault('logy', False)

    # Set default title labels
    cfg.setdefault('xtitle',   '')
    cfg.setdefault('ytitle',   'Events')
    cfg.setdefault('topLeft',  '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}')
    cfg.setdefault('topRight', f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{-1}}')

    # Configure ratio plot settings if enabled
    if (('ymin' not in cfg) or ('ymax' not in cfg)) and ratio_plot:
        LOGGER.error('Ratio limits of the histogram not set. Aborting code')
        exit(1)
    cfg.setdefault('ytitleR', 'Ratio')
    cfg.setdefault('ratiofraction', 0.3)

    return cfg


def define_legend(
        num_entries: int,
        columns: int = 1,
        x1: float = 0.55,
        y1: float = 0.99,
        x2: float = 0.99,
        y2: float = 0.90,
        border_size: int = 0,
        fill_style: int = 0,
        text_size: float = 0.03,
        set_margin: float = 0.2,
        text_font: int = -1
) -> Any:
    '''Create the legend used by the plot.'''

    import ROOT
    leg = ROOT.TLegend(x1, y1 - num_entries * 0.06 / columns, x2, y2)

    if text_font != -1:
        leg.SetTextFont(text_font)
    leg.SetBorderSize(border_size)
    leg.SetFillStyle(fill_style)
    leg.SetTextSize(text_size)
    leg.SetMargin(set_margin)
    leg.SetNColumns(columns)

    return leg


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
    latex.SetTextAlign(text_align)
    latex.SetTextColor(text_color)
    latex.SetTextFont(text_font)
    return latex


def draw_latex(
    latex: ROOT.TLatex,
    text_data: list[tuple[str, float, float, float]]
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

    return None
