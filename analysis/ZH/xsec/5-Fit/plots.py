#!/usr/bin/env python3
"""
Simplified 1D likelihood scan plotter using matplotlib and uproot.
Optimized for speed and simplicity while maintaining plot1DScan.py structure.
"""

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import time, uproot

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import UnivariateSpline

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,
    allow_empty=True,
    ecm_multi=True,
    include_sels=True,
    fit_plot=True,
    description='Fit Plots Script'
)
arg = parse_args(parser, False, False)
set_log(arg)

LOGGER = get_logger(__name__)



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from package.userConfig import loc, plot_file
loc.set_default_type(Path)
from package.config import timer, H_decays, H_labels
from package.plots.python.plotter import (
    set_plt_style,
    set_labels,
    savefigs
)



################################
### MATPLOTLIB STYLE SETTING ###
################################

set_plt_style()
plt.ioff()



####################
### CONFIG SETUP ###
####################

# Parse main inputs
cats, ecms, sels = arg.cat.split('-'), arg.ecm.split('-'), arg.sels.split('-')
if arg.lep:     cats.append('leptonic')
if arg.combine: cats.append('combined')

params_label = {
    'ee':       r'$Z(\to e^+e^-)H$',
    'mumu':     r'$Z(\to\mu^+\mu^-)H$',
    'qq':       r'$Z(\to q\bar{q})H$',
    'leptonic': r'$Z(\to\ell^+\ell^-)H$',
    'combined': r'$ZH$',
    '240':      '240 GeV',
    '365':      '365 GeV',
}



##########################
### PLOTTING FUNCTIONS ###
##########################

def read_tree_data(
        file_path: Path,
        param: str = "r",
        y_cut: float | int = 7.0
         ) -> tuple[np.ndarray,
                    np.ndarray]:
    """
    Read limit tree from ROOT file using uproot.
    Returns sorted parameter and 2*deltaNLL arrays.
    """
    with uproot.open(file_path) as f:
        tree = f["limit"]
        # Read data directly as numpy arrays
        x_data = tree[param].array(library="np")
        y_data = 2 * tree["deltaNLL"].array(library="np")

    LOGGER.debug(f'Found {len(x_data)} point in {file_path}')

    # Create combined array, sort, and remove duplicates
    data = np.column_stack([x_data, y_data])
    data = data[data[:, 0].argsort()]  # Sort by x
    data = data[np.diff(data[:, 0], prepend=-np.inf) > 1e-8]  # Remove x duplicates
    LOGGER.debug(f'Removed {len(x_data)-len(data)} duplicate, {len(data)} points remaining')

    # Remove points above y cutoff
    data = data[data[:, 1] <= y_cut]
    LOGGER.debug(f'Removed points greater than {y_cut = }, {len(data)} points remaining')

    return data[:, 0], data[:, 1]


def find_crossings(
        x_sorted: np.ndarray,
        spl: UnivariateSpline,
        y_val: float | int,
        tol: float | int = 1e-4
         ) -> list[dict[str, float | int]]:
    """
    Find crossing intervals of spline at given y value using robust bisection method.
    Returns list of dicts with 'lo', 'hi' boundaries of each confidence interval.
    """
    crossing_points = []
    x_vals = np.linspace(x_sorted[0], x_sorted[-1], 1000)
    y_vals = spl(x_vals)

    # Find regions where spline crosses y_val (sign changes)
    diff = y_vals - y_val
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    # For each sign change, find the exact crossing point using bisection
    for idx in sign_changes:
        x_lo, x_hi = x_vals[idx], x_vals[idx + 1]

        # Bisect to find more accurate crossing point
        for _ in range(20):
            x_mid = (x_lo + x_hi) / 2
            if abs(spl(x_mid) - y_val) < tol:
                break
            if (spl(x_lo) - y_val) * (spl(x_mid) - y_val) < 0:
                x_hi = x_mid
            else:
                x_lo = x_mid

        # Store the converged crossing point
        crossing_points.append((x_lo + x_hi) / 2)

    # Pair up consecutive crossings into intervals (left boundary, right boundary)
    crossings = []
    for i in range(0, len(crossing_points) - 1, 2):
        crossings.append({'lo': crossing_points[i], 'hi': crossing_points[i + 1]})

    # If odd number of crossings, pair the last one with itself (edge case)
    if len(crossing_points) % 2 == 1:
        last = crossing_points[-1]
        crossings.append({'lo': last, 'hi': last})

    return crossings


def process_scan(
        input_file: Path,
        param: str = "r",
        y_cut: float | int = 7.0
         ) -> tuple[np.ndarray,
                    np.ndarray,
                    int | float,
                    UnivariateSpline,
                    tuple[float | int, float | int],
                    tuple[float | int, float | int]]:
    """
    Process a single likelihood scan and return results.
    Returns: (x, y, bestfit, spline, errors_1sig, errors_2sig)
    """
    # Read data
    x, y = read_tree_data(input_file, param=param, y_cut=y_cut)

    if len(x) <= 1:
        raise ValueError(f"Not enough points in {input_file}: {len(x)}")

    # Find best fit: prefer exact y=0 if exists, otherwise use minimum
    zero_idx = np.where(y == 0)[0]
    bestfit = x[zero_idx[0]] if len(zero_idx) > 0 else x[np.argmin(y)]

    # Fit spline with small smoothing to preserve data features
    spl = UnivariateSpline(x, y, k=3, s=0.01)

    # Find crossings at all y values
    crossings = {yv: find_crossings(x, spl, yv) for yv in [1.0, 4.0]}

    # Calculate errors
    def get_error_band(
            crossings_list: dict,
            bf: float | int
             ) -> tuple[float | int, float | int]:
        """Find crossing containing best fit and calculate error."""
        for cr in crossings_list:
            if bf >= cr['lo'] and bf <= cr['hi']:
                err_hi = cr['hi'] - bf
                err_lo = bf - cr['lo']
                return err_hi, err_lo
        return 0.0, 0.0

    err_hi1, err_lo1 = get_error_band(crossings[1.0], bestfit)
    err_hi2, err_lo2 = get_error_band(crossings[4.0], bestfit)

    return x, y, bestfit, spl, (err_hi1, err_lo1), (err_hi2, err_lo2)


def plot_1d_scans(
        input_specs: list[tuple[str, ...]],
        output: Path,
        param: str = 'r',
        y_cut: int | float = 7.0,
        y_max: int | float = -1,
        suffix: str = '',
        right: str = '',
        sig2: bool = False) -> None:
    """
    Plot multiple 1D likelihood scans on the same figure.

    Args:
        input_specs: List of tuples (filepath, label, color) or list of filepaths
        output: Output filename without extension
        param: Parameter name to scan
        y_cut: Remove points with y > y_cut
        y_max: Y-axis maximum
        suffix: Suffix to put after output file name
        right: Title to put at the right of the figure
        sig2: include 2 sigma uncertainty in the plot
    """
    LOGGER.info(f'Processing {len(input_specs)} scan(s): {output}')

    # Normalize input format
    scans_data = []
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, spec in enumerate(input_specs):
        if isinstance(spec, tuple):
            filepath, label, color = spec
        else:
            filepath, label = spec, f"Scan {i+1}"
            color = colors[i % len(colors)]

        x, y, bestfit, spl, err_1sig, err_2sig = process_scan(filepath, param, y_cut)
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = spl(x_smooth)

        scans_data.append({
            'filepath': filepath,
            'label':    params_label.get(label, label),
            'color':    color,
            'x':        x,
            'y':        y,
            'x_smooth': x_smooth,
            'y_smooth': y_smooth,
            'bestfit':  bestfit,
            'spl':      spl,
            'err_1sig': err_1sig,
            'err_2sig': err_2sig,
        })

    # Create figure
    fig, ax = plt.subplots(dpi=300)

    try:

        # Plot all scans
        for scan in scans_data:
            ax.scatter(scan['x'], scan['y'], s=15, color=scan['color'], zorder=2)
            ax.plot(scan['x_smooth'], scan['y_smooth'], '-', linewidth=2.5,
                    color=scan['color'], label=scan['label'], zorder=3)
            for yv in [1.0, 4.0]:
                crossings_yv = find_crossings(scan['x'], scan['spl'], yv)
                for cr in crossings_yv:
                    ax.plot([cr['lo'], cr['lo']], [0, yv], color=scan['color'], alpha=0.5, linestyle='dashed')
                    ax.plot([cr['hi'], cr['hi']], [0, yv], color=scan['color'], alpha=0.5, linestyle='dashed')

        # Draw horizontal reference lines for 1σ and 2σ
        ax.axhline(1.0,   color='gray', linestyle='dashed')
        ax.axhline(4.0,   color='gray', linestyle='dashed')
        ax.axhline(y_cut, color='black')

        # Formatting
        set_labels(ax, param, r'$-2\Delta\ln\mathcal{L}$', right=right)

        # Calculate required y_max based on data and text
        max_y_smooth = max(np.max(scan['y_smooth']) for scan in scans_data)
        num_text_lines = sum(1 + (1 if sig2 else 0) for _ in scans_data)
        text_box_height = num_text_lines * 0.7
        y_max_required = max_y_smooth + text_box_height + 0.5   # +0.5 margin
        effective_y_max = max(y_max, y_max_required)

        ax.set_ylim(0, effective_y_max)

        # Set x-axis limits based on all scans
        all_x = np.concatenate([s['x'] for s in scans_data])
        margin = 0.05 * (all_x.max() - all_x.min())
        ax.set_xlim([all_x.min() - margin, all_x.max() + margin])

        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=15)

        # Add text box with fit results (all scans)
        textstr = ""
        for i, scan in enumerate(scans_data):
            err_hi_1, err_lo_1 = scan['err_1sig']
            err_hi_2, err_lo_2 = scan['err_2sig']
            textstr += f"{scan['label']}: {param} = {scan['bestfit']:.3f}$_{{-{err_lo_1*100:.2f}\\%}}^{{+{err_hi_1*100:.2f}\\%}}$ (68\\%)"
            if sig2:
                textstr += f"\n{' '*len(scan['label'])}: {param} = {scan['bestfit']:.3f}$_{{-{err_lo_2*100:.2f}\\%}}^{{+{err_hi_2*100:.2f}\\%}}$ (95\\%)"
            if i < len(scans_data) - 1:
                textstr += "\n"

        props = dict(boxstyle='round', facecolor='white', pad=0.8)
        ax.text(0.975, 0.96, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, fontsize=15, family='monospace')

        # Save outputs
        savefigs(fig, output, 'Scan', '_'+suffix,plot_file)

    finally:
        plt.close(fig)



##########################
### EXECUTION FUNCTION ###
##########################

def main():

    colors  = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    if arg.bias:
        # Individual plots for all combinations
        targets = H_decays if 'all' in arg.targets else arg.target.split('-')
        for cat, ecm, sel in [(c, e, s) for c in cats for e in ecms for s in sels]:
            all_scans = []
            for i, target in enumerate(targets):
                inDir  = loc.get('BIAS_WS',     cat, ecm, sel)
                outDir = loc.get('BIAS_RESULT', cat, ecm, sel)
                scan   = (inDir / f'higgsCombineXsec{target}.MultiDimFit.mH125.root',
                          H_labels.get(target, target),
                          colors[i % len(colors)])
                all_scans.append(scan)
            plot_1d_scans(all_scans, outDir, arg.param, arg.y_cut, arg.y_max)

    else:
        if arg.which == '':
            # Individual plots for all combinations
            for cat, ecm, sel in [(c, e, s) for c in cats for e in ecms for s in sels]:
                inDir  = loc.get('NOMINAL_WS',     cat, ecm, sel)
                outDir = loc.get('NOMINAL_RESULT', cat, ecm, sel)
                scan = (inDir / 'higgsCombineXsec.MultiDimFit.mH125.root', "Observed", colors[0])
                plot_1d_scans([scan], outDir, arg.param, arg.y_cut, arg.y_max)
        else:
            # Comparison plot: configure varying parameter and fixed values
            from itertools import product

            config = {
                'cat': (cats, [ecms, sels]),
                'ecm': (ecms, [cats, sels]),
                'sel': (sels, [cats, ecms])
            }
            varying, fixed_lists = config[arg.which]
            insert_pos = ['cat', 'ecm', 'sel'].index(arg.which)

            for fixed_vals in product(*fixed_lists):
                fixed_str = '_'.join(str(f) for f in fixed_vals)
                param_label = ', '.join(params_label.get(str(f), f) for f in fixed_vals)

                all_scans = []
                for i, var_val in enumerate(varying):
                    param = list(fixed_vals)
                    param.insert(insert_pos, var_val)

                    inDir = loc.get('NOMINAL_WS', *param)
                    scan  = (inDir / 'higgsCombineXsec.MultiDimFit.mH125.root',
                             str(var_val),
                             colors[i % len(colors)])
                    all_scans.append(scan)

                out: Path = loc.get('PLOTS_FIT_SCAN') / arg.which
                out.mkdir(exist_ok=True, parents=True)
                plot_1d_scans(
                    all_scans, out, arg.param,
                    arg.y_cut, arg.y_max,
                    fixed_str, param_label,
                    arg.sig2
                )


##########################
### EXECUTION FUNCTION ###
##########################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
