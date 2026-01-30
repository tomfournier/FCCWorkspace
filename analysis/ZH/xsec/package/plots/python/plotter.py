'''Matplotlib plotting utilities and styling.

Provides:
- Global matplotlib style configuration: `set_plt_style()`.
- Axis labeling with FCC-ee branding: `set_labels()`.
- Multi-format figure export: `savefigs()`.
- Consistent typography and appearance across all matplotlib-based plots.

Functions:
- `set_plt_style()`: Configure matplotlib RC parameters for serif fonts, figure size, and grid styling.
- `set_labels()`: Set axis labels and titles with FCC-ee simulation watermark and optional metadata.
- `savefigs()`: Export matplotlib figures to disk in PNG, PDF, or other formats with automatic path creation.

Conventions:
- Global matplotlib style applied once via `set_plt_style()` before creating plots.
- All plots include FCC-ee simulation watermark in top-left corner (customizable via `left` parameter).
- Axis labels positioned at plot edges ('right' for x-axis, 'top' for y-axis by default).
- Figure size fixed at 12x8 inches (1440x960 pixels at 120 DPI).
- Font configuration uses serif family (Roman) with 30pt default and 25pt for axes.
- LaTeX text rendering enabled for mathematical symbols and special formatting.
- Grid enabled by default for easier value reading.

Usage:
- Initialize matplotlib styling once per session with `set_plt_style()`.
- Decorate all figures with `set_labels()` to add axis titles and experiment branding.
- Export finished plots to multiple formats simultaneously with `savefigs()`.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import matplotlib.pyplot as plt



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

#_______________________
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
    
    Args:
        ax (Axes): Matplotlib axes object to configure.
        xlabel (str, optional): Label for x-axis. Defaults to ''.
        ylabel (str, optional): Label for y-axis. Defaults to ''.
        left (str, optional): Title for left side of plot. Defaults to '' (FCC-ee simulation label).
        right (str, optional): Title for right side of plot. Defaults to ''.
        locx (str, optional): Horizontal position for xlabel. Defaults to 'right'.
        locy (str, optional): Vertical position for ylabel. Defaults to 'top'.
    
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

#______________________________
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
        print(f"\tSaved {plotname.replace('_', ' ')} "
              f'plot to {outDir}/{plotname}{suffix}.{f}')