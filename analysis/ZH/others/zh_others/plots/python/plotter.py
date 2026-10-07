'''Matplotlib plotting utilities and styling for FCC-ee analysis.

Provides consistent plotting interface with global style configuration, FCC-ee branding,
and multi-format figure export for publication-ready plots.

Functions:
- `set_plt_style()`: Configure matplotlib RC parameters globally once per session.
- `set_labels(ax, xlabel, ylabel, left, right, locx, locy)`: Decorate axes with labels and FCC-ee watermark.
- `savefigs(fig, outDir, plotname, suffix, format)`: Export figure to disk in one or multiple formats.

Matplotlib Configuration:
- Font: Serif family (Roman) with 30pt base size
- Figure size: 12x8 inches (1440x960 pixels at 120 DPI)
- Axes: 25pt title, 30pt labels; grid enabled for readability
- Ticks: 25pt label size for both x and y axes
- Legend: 14pt font size
- LaTeX: Text rendering enabled for math symbols and special formatting

Conventions:
- Call `set_plt_style()` once at the start of your plotting session.
- Use `set_labels()` on every figure to add FCC-ee Simulation watermark and metadata.
- Label positioning: xlabel at 'right', ylabel at 'top' by default for clean layout.
- FCC-ee branding appears in top-left corner (customizable or removable via left parameter).
- Supports LaTeX math mode in all labels and titles (e.g., r'$m_{ll}$ [GeV]').
- Save figures in multiple formats simultaneously (png, pdf, etc.) for different use cases.

Usage Examples:
- Session setup: set_plt_style()
- Create plot: fig, ax = plt.subplots()
- Add decoration: set_labels(ax, r'$p_T$ [GeV]', 'Events', right='240 GeV, 5 ab$^{-1}$')
- Save plot: savefigs(fig, './plots', 'signal_mass', suffix='_ee', format=['png', 'pdf'])
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from ...logger import get_logger

LOGGER = get_logger(__name__)



######################
### MAIN FUNCTIONS ###
######################

# ___________________________
def set_plt_style() -> None:
    '''Configure matplotlib plotting style with consistent font and figure settings.

    Sets global matplotlib RC parameters including serif fonts, figure size,
    and tick/label sizes for consistent plot appearance across the analysis.

    Returns:
        None
    '''
    from matplotlib import rc

    # Font configuration: serif with 30pt size
    rc('font', **{'family': 'serif',
                  'serif': ['Roman'],
                  'size': 30})
    # Figure dimensions in inches (width, height)
    rc('figure', figsize=(12, 8))
    # Axes labels and grid configuration
    rc('axes', titlesize=25,
       labelsize=30, grid=True)
    # Tick label sizes for both axes
    rc('xtick', labelsize=25)
    rc('ytick', labelsize=25)
    # Legend and text rendering settings
    rc('legend', fontsize=14)
    rc('text', usetex=True)

# _______________________
def set_labels(
    ax: 'plt.Axes',
    xlabel: str = '',
    ylabel: str = '',
    left: str = '',
    right: str = '',
    locx: str = 'right',
    locy: str = 'top'
     ) -> None:
    '''Configure axis labels and titles with FCC-ee experiment branding.

    Sets x-axis, y-axis, and title labels on a matplotlib axes with FCC-ee
    branding in the left corner. By default, adds "FCC-ee Simulation" watermark
    unless explicitly removed. Supports positioning labels at plot edges.

    Args:
        ax (Axes): Matplotlib axes object to configure.
        xlabel (str, optional): Label for x-axis. Defaults to ''.
        ylabel (str, optional): Label for y-axis. Defaults to ''.
        left (str, optional): Title for left side of plot. If empty string (default),
               displays "FCC-ee Simulation" watermark. Pass 'None' or ' ' to remove
               the left title entirely. Defaults to ''.
        right (str, optional): Title for right side of plot (e.g., luminosity info). Defaults to ''.
        locx (str, optional): Horizontal position for xlabel: 'left', 'center', or 'right'. Defaults to 'right'.
        locy (str, optional): Vertical position for ylabel: 'bottom', 'center', or 'top'. Defaults to 'top'.

    Returns:
        None
    '''
    # Default FCC-ee simulation label if not specified
    if not left:
        left = r'$\textbf{FCC-ee \textit{Simulation}}$'
    # Allow explicit removal of left label
    elif (left=='None') or (left==' '):
        left = ''

    ax.set_xlabel(xlabel, loc=locx)
    ax.set_ylabel(ylabel, loc=locy)
    ax.set_title(left,    loc='left')
    ax.set_title(right,   loc='right')

# ______________________________
def savefigs(
    fig: 'plt.Figure',
    outDir: str,
    plotname: str,
    suffix: str = '',
    format: list[str] = ['png']
     ) -> None:
    '''Save figure to disk in one or multiple formats.

    Args:
        fig (Figure): Matplotlib figure object to save.
        outDir (str): Output directory path.
        plotname (str): Base name for the output file.
        suffix (str, optional): Suffix to append to plotname. Defaults to ''.
        format (list[str], optional): List of file formats to save (e.g., ['png', 'pdf']). Defaults to ['png'].

    Returns:
        None
    '''
    from os.path import join

    # Construct full output file path without extension
    fpath = join(outDir,f'{plotname}{suffix}')
    # Save figure in each specified format
    for f in format:
        fig.savefig(f'{fpath}.{f}', bbox_inches='tight')
        LOGGER.info(f'Saved plot to {fpath}.{f}')
