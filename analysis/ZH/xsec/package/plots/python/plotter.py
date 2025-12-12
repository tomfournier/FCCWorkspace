from os.path import join
from matplotlib import rc
from matplotlib.pyplot import Figure, Axes

#___________________________
def set_plt_style() -> None:
    rc('font', **{'family': 'serif', 
                  'serif': ['Roman'], 
                  'size': 30})
    rc('figure', figsize=(12, 8))
    rc('axes', titlesize=25, 
       labelsize=30, grid=True)
    rc('xtick', labelsize=25)
    rc('ytick', labelsize=25)
    rc('legend', fontsize=14)
    rc('text', usetex=True)

#_________________________________________
def set_labels(ax: Axes, 
               xlabel: str = '', 
               ylabel: str = '', 
               left: str = '', 
               right: str = '', 
               locx: str = 'right', 
               locy: str = 'top') -> None:
    if not left:
        left = r'$\textbf{FCC-ee \textit{Simulation}}$'
    elif (left=='None') or (left==' '):
        left = ''

    ax.set_xlabel(xlabel, loc=locx)
    ax.set_ylabel(ylabel, loc=locy)
    ax.set_title(left,    loc='left')
    ax.set_title(right,   loc='right')

#_______________________________________
def savefigs(fig: Figure, 
             outDir: str, 
             plotname: str, 
             suffix: str = '', 
             format: list[str] = ['png']
             ) -> None:
    fpath = join(outDir,f'{plotname}{suffix}')
    for f in format:
        fig.savefig(f'{fpath}.{f}', bbox_inches='tight')
        print(f"\tSaved {plotname.replace('_', ' ')} "
              f'plot to {outDir}/{plotname}{suffix}.{f}')