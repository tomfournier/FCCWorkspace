####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import os, uproot

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

from logger import get_logger, INSPECT, INSP_DBG

LOGGER = get_logger(__name__)




###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from path import PathObj
from plot.pyplot import MatplotlibPlot
from plot.pyplotter import set_plt_style



################################
### MATPLOTLIB STYLE SETTING ###
################################

set_plt_style()
plt.ioff()



############################
### VARIABLES DEFINITION ###
############################

params_label = {
    'ee':       r'$Z(\to e^+e^-)H$',
    'mumu':     r'$Z(\to\mu^+\mu^-)H$',
    'qq':       r'$Z(\to q\bar{q})H$',
    'leptonic': r'$Z(\to\ell^+\ell^-)H$',
    'combined': r'$ZH$',
    '240':      '240 GeV',
    '365':      '365 GeV',
    '':         '240-365 GeV'
}



########################
### HELPER FUNCTIONS ###
########################

def parse_sels(sel: str | None, cat: str, combined: bool) -> tuple[dict[str, str], str]:
    '''
    Parse the selection argument, with optional per-channel overrides.

    Syntax: positional ('A' for all, or 'A,B,C' for ee,mumu,qq) and/or 'cat=sel'.
    Per-channel selections are only allowed when combining channels.

    Returns:
        tuple: Selection per channel, and the selection used for the shared paths.
    '''

    chans, tokens = ('ee', 'mumu', 'qq'), (sel or '').split(',')
    pos = [t for t in tokens if '=' not in t]
    sels = dict(zip(chans, pos if len(pos) > 1 else pos * 3)) | dict(t.split('=') for t in tokens if '=' in t)

    if set(sels) != set(chans) or (len(tokens) > 1 and not combined):
        raise ValueError(f"Need a selection for each of {chans} (per-channel only with lep/combine), got {sels}")
    if combined and len(set(sels.values())) > 1:
        LOGGER.log(INSPECT, 'Using per-channel selections:\n\t' + '\n\t'.join(f"{cat:<5}: {sel}" for cat, sel in sels.items()))
    return sels, sels.get(cat, sels['ee'])


def add_stamp(
        path: PathObj,
        label: str,
        status: str = 'ok'
         ) -> str:
    '''Add a timestamped stamp to a log file for tracking execution status.'''
    from uuid import uuid4
    from datetime import datetime

    # Generate timestamp and unique ID
    ts = datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
    uniq = uuid4().hex[:8]

    # Append stamp to log file
    with open(path, 'a') as log_file:
        log_file.write(f'\n\n---- STAMP {label} [{status}]: {ts} | id={uniq}\n')
    LOGGER.log(INSPECT, f'Added STAMP: {ts} | id={uniq} | status={status} | file={label}')
    return uniq


def run_cmd(
        cmd: list[str] | str,
        log_file: Path | PathObj | str | None = None,
        cwd: Path | PathObj | str | None = None,
        env: os._Environ | None = None,
        return_stamp: bool = False
) -> int | tuple[int, str]:
    '''Run a command using subprocess'''
    import subprocess

    if isinstance(log_file, str): log_file = PathObj(log_file)
    status, stamp = 'not-run', None
    try:
        if log_file is not None:
            with open(log_file, 'w') as out:
                result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=cwd, env=env, check=True)
        else:
            result = subprocess.run(cmd, text=True, capture_output=False, cwd=cwd, env=env, check=True)
        status = 'ok'
    except subprocess.CalledProcessError as exc:
        if status == 'not-run': status = f'error exit = {exc.returncode}'
        LOGGER.error(f'Command failed with exit code {exc.returncode}', exc_info=True)
        if log_file is not None and log_file.exists():
            LOGGER.error(f'The log file is at {log_file}')
        return exc.returncode
    finally:
        if log_file is not None and log_file.exists():
            stamp = add_stamp(log_file, log_file.name, status)

    if return_stamp:
        return result.returncode, stamp
    return result.returncode


def get_grid_number(n: int, params: list[str] | int, pm: bool = True) -> int:
    if n % 2 == 0:
        n = n+1 if pm else n-1
    if isinstance(params, int):
        return n**params
    return n**len(params)


def check_log(res_log: str) -> int | None:
    '''Check if the fit has converged'''
    LOGGER.debug('Fit done, checking fit log')
    status = None
    with open(str(res_log)) as file:
        lines = file.readlines()
    for line in reversed(lines):
        if 'WARNING: MultiDimFit failed' in line:
            status = -1
            break
    if status is None:
        for line in reversed(lines):
            if 'Warning - No valid high-error found' in line or \
                    'Warning - No valid low-error found' in line:
                status = -2

    if status is None:
        LOGGER.debug(f'Minimization success, {status = }')
    elif status == -1:
        LOGGER.error(f'Minimization did not converge\nCheck the log at {res_log}')
    elif status == -2:
        LOGGER.error(f"Couldn't find the high/low-error\nCheck the log at {res_log}")
    return status


def get_results(file, params, algo) -> dict[str, float | int]:

    res = {}
    if isinstance(params, str): params = [params]

    if algo == 'singles':
        tree: dict[str, np.ndarray] = uproot.open(file)['limit'].arrays(params, library='np')
        for i, param in enumerate(params):
            bestfit = float(tree[param][0])
            if bestfit < 0 and bestfit > -1e-2: bestfit = abs(bestfit)
            err_lo, err_hi = abs(float(tree[param][2 * i + 1]) - bestfit), abs(float(tree[param][2 * i + 2]) - bestfit)
            res[param] = {'best_fit': bestfit, 'err_lo': err_lo, 'err_hi': err_hi, 'err': (err_hi + err_lo) / 2}

    elif algo == 'grid':
        for param in params:
            other_params = [p for p in params if p!=param] if len(params)>1 else []
            bestfit, err_hi, err_lo = process_scan(file, param, 1e3, True, other_params)
            res[param] = {'best_fit': bestfit, 'err_lo': err_lo, 'err_hi': err_hi, 'err': (err_hi + err_lo) / 2}
    else:
        raise ValueError(f'{algo = } is not supported, choose between [singles, grid]')

    return res


def save_results(
        results: dict[str, float | int],
        outDir: str,
        print_result: bool = True,
        target: str = '',
) -> None:
    '''Save fitting results (signal strength and uncertainty) to output file.'''

    import json

    if not results:
        LOGGER.warning("Couldn't extract values of the fit, go to the log file to have more information")
        return None

    unc_msg = ''

    msg = 'Results successfully extracted:'
    for param, res in results.items():
        precision = 6 if param == 'r' else 2
        msg += f'\n\t{param} = {res["best_fit"]:.{precision}f} +/- {res["err"]:.{precision}f}' if print_result else ''
        if param == 'r': unc_msg += f'Uncertainty obtained on ZH cross-section: {res["err"]:.2%}'
    LOGGER.info(msg)
    if unc_msg and print_result: LOGGER.info(unc_msg)
    out_file = Path(outDir) / f'results{target}.json'
    out_file.write_text(json.dumps(results, indent=4))

    LOGGER.debug(f'Saved results in {out_file}')

    return None



#################################
### DATA PROCESSING FUNCTIONS ###
#################################

def read_tree_data(
        input_file: Path,
        param: str = "r",
        y_cut: float | int = 7.0,
        other_params: list[str] | None = None,
        eps: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read limit tree from ROOT file using uproot.
    Returns sorted parameter and 2*deltaNLL arrays.
    """
    branches = [param, "deltaNLL", *(other_params or [])]
    with uproot.open(input_file) as f:
        tree: dict[str, np.ndarray] = f["limit"].arrays(branches, library='np')

        if other_params:
            mask = np.ones(len(tree[param]), dtype=bool)
            for other_param in other_params:
                mask &= np.abs(tree[other_param]) <= eps
            x_data = tree[param][mask]
            y_data = 2 * tree["deltaNLL"][mask]
        else:
            x_data = tree[param]
            y_data = 2 * tree["deltaNLL"]

    LOGGER.log(INSPECT, f'Found {len(x_data)} point in {input_file}')

    # Sort by x and remove near-duplicate x values without combining columns.
    order = np.argsort(x_data)
    x_data, y_data = x_data[order], y_data[order]
    keep = np.diff(x_data, prepend=-np.inf) > 1e-8
    x_data, y_data = x_data[keep], y_data[keep]
    LOGGER.log(INSPECT, f'Removed {len(order)-len(x_data)} duplicate, {len(x_data)} points remaining')

    # Remove points above y cutoff
    keep = y_data <= y_cut
    x_data, y_data = x_data[keep], y_data[keep]
    LOGGER.log(INSPECT, f'Removed points greater than {y_cut = }, {len(x_data)} points remaining')

    return x_data, y_data


def find_crossings(
        x_sorted: np.ndarray,
        spl: UnivariateSpline,
        y_val: float | int,
        tol: float | int = 1e-4
         ) -> list[dict[str, float | int]]:
    """
    Find crossing intervals of spline at given y value using a bracketed root finder.
    Returns list of dicts with 'lo', 'hi' boundaries of each confidence interval.
    tol is the absolute x-coordinate tolerance used to refine each crossing.
    """
    crossing_points = []
    x_vals = np.linspace(x_sorted[0], x_sorted[-1], 1000)
    y_vals = spl(x_vals)

    # Find regions where spline crosses y_val (sign changes)
    diff = y_vals - y_val
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    # Refine each bracket with Brent's method rather than fixed-iteration bisection.
    for idx in sign_changes:
        crossing_points.append(brentq(lambda x: spl(x) - y_val, x_vals[idx], x_vals[idx + 1], xtol=tol))

    # Pair up consecutive crossings into intervals (left boundary, right boundary)
    crossings = []
    for i in range(0, len(crossing_points) - 1, 2):
        crossings.append({'lo': crossing_points[i], 'hi': crossing_points[i + 1]})

    # If odd number of crossings, pair the last one with itself (edge case)
    if len(crossing_points) % 2 == 1:
        last = crossing_points[-1]
        crossings.append({'lo': last, 'hi': last})

    return crossings


def _analyze_scan(
        input_file: Path,
        param: str,
        y_cut: float | int,
        other_params: list[str],
        eps: float
         ) -> tuple[np.ndarray, np.ndarray, float, UnivariateSpline,
                    dict[float, list[dict[str, float | int]]],
                    tuple[float | int, float | int], tuple[float | int, float | int]]:
    x, y = read_tree_data(input_file, param, y_cut, other_params, eps)
    if len(x) <= 1:
        raise ValueError(f"Not enough points in {input_file}: {len(x)}")

    zero_idx = np.flatnonzero(y == 0)
    bestfit = float(x[zero_idx[0]] if len(zero_idx) else x[np.argmin(y)])
    spl = UnivariateSpline(x, y, k=3, s=0.01)
    levels = (1.0, 4.0)
    crossings = {level: find_crossings(x, spl, level) for level in levels}

    def error_band(level: float) -> tuple[float, float]:
        for crossing in crossings[level]:
            if crossing['lo'] <= bestfit <= crossing['hi']:
                return crossing['hi'] - bestfit, bestfit - crossing['lo']
        return 0.0, 0.0

    err_1sig, err_2sig = error_band(1.0), error_band(4.0)
    LOGGER.log(INSPECT,  f'Found best fit for {param} at {bestfit}')
    LOGGER.log(INSPECT,  f'1 sigma error found at +{err_1sig[0]:.2g}/-{err_1sig[1]:.2g}')
    LOGGER.log(INSP_DBG, f'2 sigma error found at +{err_2sig[0]:.2g}/-{err_2sig[1]:.2g}')
    return x, y, bestfit, spl, crossings, err_1sig, err_2sig


def process_scan(
        input_file: Path,
        param: str = "r",
        y_cut: float | int = 7.0,
        only_res: bool = False,
        other_params: list[str] = [],
        eps: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, int | float, UnivariateSpline,
           tuple[float | int, float | int], tuple[float | int, float | int]]:
    """Process a likelihood scan and return the fit and uncertainty results."""
    x, y, bestfit, spl, _, err_1sig, err_2sig = _analyze_scan(input_file, param, y_cut, other_params, eps)
    if only_res:
        return bestfit, *err_1sig
    return x, y, bestfit, spl, err_1sig, err_2sig


def read_tree_data_2d(
        file_path: Path,
        x_param: str = 'Cphi',
        y_param: str = 'Cbox',
        z_cut: float | int = 1e0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a 2D likelihood scan from a ROOT file using uproot.

    Returns x, y, and 2*deltaNLL arrays with duplicate (x, y) points collapsed.
    """
    with uproot.open(file_path) as f:
        tree = f['limit']

        if x_param not in tree.keys():
            raise KeyError(f"Branch '{x_param}' not found in {file_path}")
        if y_param not in tree.keys():
            raise KeyError(f"Branch '{y_param}' not found in {file_path}")

        raw_data: dict[str, np.ndarray] = tree.arrays([x_param, y_param, 'deltaNLL'], library='np')

    x_data, y_data = raw_data[x_param], raw_data[y_param]
    z_data = 2 * raw_data['deltaNLL']
    LOGGER.log(INSPECT, f'Found {len(z_data)} point in {file_path}')

    keep = z_data <= z_cut
    x_data, y_data, z_data = x_data[keep], y_data[keep], z_data[keep]
    LOGGER.log(INSPECT, f'Removed points greater than {z_cut = }, {len(z_data)} points remaining')

    if len(z_data) == 0:
        return x_data, y_data, z_data

    coords = np.column_stack((x_data, y_data))
    uniq_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    uniq_z = np.full(len(uniq_coords), np.inf)
    np.minimum.at(uniq_z, inverse, z_data)

    LOGGER.log(INSPECT, f'Removed {len(z_data) - len(uniq_z)} duplicate, {len(uniq_z)} points remaining')

    return uniq_coords[:, 0], uniq_coords[:, 1], uniq_z


def process_scan_2d(
        input_file: Path,
        x_param: str = 'Cphi',
        y_param: str = 'Cbox',
        z_cut: float | int = 50.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Process a single 2D likelihood scan and return the scan points and best-fit coordinates.
    """
    x, y, z = read_tree_data_2d(input_file, x_param, y_param, z_cut)

    if len(x) <= 2:
        raise ValueError(f"Not enough points in {input_file}: {len(x)}")

    best_idx = z == z.min()
    best_x, best_y = float(x[best_idx]), float(y[best_idx][0])
    LOGGER.log(INSPECT, f'Found best fit at {x_param} = {best_y}, {y_param} = {best_y}')

    return x, y, z, best_x, best_y



##########################
### PLOTTING FUNCTIONS ###
##########################

def plot_1d_scans(
        input_specs: list[tuple[str, ...]],
        outputdir: Path,
        param: str = 'r',
        y_cut: int | float = 7.0,
        y_max: int | float = -1,
        suffix: str = '',
        right: str = '',
        sig2: bool = False,
        other_params: list[str] = [],
        coeffs_label: dict[str, str] = {},
        file_formats: list[str] = ['png']
) -> None:
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
    LOGGER.debug(f'Processing {len(input_specs)} scan(s): {outputdir}')

    # Normalize input format
    scans_data = []
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    plot_y_cut = min(y_max, y_cut) if 0 < y_max < y_cut else y_cut

    for i, spec in enumerate(input_specs):
        if isinstance(spec, tuple):
            filepath, label, color = spec
        else:
            filepath, label = spec, f"Scan {i+1}"
            color = colors[i % len(colors)]

        x, y, bestfit, spl, crossings, err_1sig, err_2sig = _analyze_scan(
            filepath, param, y_cut, other_params, 1e-5)
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = spl(x_smooth)
        smooth_mask, data_mask = y_smooth <= plot_y_cut, y <= plot_y_cut
        x_smooth, y_smooth = x_smooth[smooth_mask], y_smooth[smooth_mask]
        x, y = x[data_mask], y[data_mask]

        scans_data.append({
            'filepath': filepath,
            'label': coeffs_label[label], 'color': color,
            'x': x, 'y': y,
            'x_smooth': x_smooth, 'y_smooth': y_smooth,
            'bestfit':  bestfit, 'crossings': crossings,
            'err_1sig': err_1sig, 'err_2sig': err_2sig,
        })

    # Create figure
    plot = MatplotlibPlot(outputdir, None, None, 'Scan', suffix, file_formats, dpi=300, style=False)
    ax = plot.axes

    try:

        plot_specs = []
        for scan in scans_data:
            x, x_smooth = scan['x'], scan['x_smooth']
            if param == 'r':
                x, x_smooth = (x - 1) * 100, (x_smooth - 1) * 100
            plot_specs.extend([
                ([x,        scan['y'],     'scatter'], {'color': scan['color'], 'zorder': 2, 's': 15}),
                ([x_smooth, scan['y_smooth'], 'plot'], {'color': scan['color'], 'zorder': 3, 'linewidth': 2.5, 'label': scan['label']}),
            ])
            for yv, crossings_yv in scan['crossings'].items():
                for cr in crossings_yv:
                    edges = [cr['lo'], cr['hi']]
                    if param == 'r': edges = [(edge - 1) * 100 for edge in edges]
                    plot_specs.extend((([[edge, edge], [0, yv], 'plot'], {'color': scan['color'], 'alpha': 0.5, 'linestyle': 'dashed'})
                                       for edge in edges))

        plot_specs.extend([
            ([1.0, 'axhline'],        {'color': 'gray', 'linestyle': 'dashed'}),
            ([4.0, 'axhline'],        {'color': 'gray', 'linestyle': 'dashed'}),
            ([plot_y_cut, 'axhline'], {'color': 'black'}),
        ])
        plot.draw(plot_specs)


        # Formatting
        lab = coeffs_label.get(param, param)
        if param == 'r':
            plot.set_labels(r'$\mu_{ZH}-1$ [\%]', r'$-2\Delta\ln\Lambda$', right=right)
        else:
            plot.set_labels(lab, r'$-2\Delta\ln\Lambda$', right=right)

        # Calculate required y_max based on data and text
        max_y_smooth = max([np.max(scan['y_smooth']) for scan in scans_data]) + \
            ([min([y_max, y_cut])] if y_max>0 else [y_max, y_cut])
        num_text_lines = sum(1 + (1 if sig2 else 0) for _ in scans_data)
        text_box_height = num_text_lines * 1.1
        y_max_required = max_y_smooth + text_box_height + 0.5
        effective_y_max = max(y_max, y_max_required)

        # Set x-axis limits based on all scans
        x_min = min(scan['x_smooth'].min() for scan in scans_data)
        x_max = max(scan['x_smooth'].max() for scan in scans_data)
        margin = 0.05 * (x_max - x_min)

        if param == 'r':
            xlim = ((x_min - margin - 1)*100, (x_max + margin - 1)*100)
        else:
            xlim = (x_min - margin, x_max + margin)

        plot.configure(xlim, (0, effective_y_max), legend=True,)
                    #    legend_kwargs={'loc': 'upper left', 'fontsize': 22})
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        def result_line(scan: dict, errors: tuple, confidence: int, show_label: bool = True) -> str:
            scale = 100 if param == 'r' else 1
            label = scan['label'] if show_label else ' ' * len(scan['label'])
            value = f"{label}: {lab} = {scan['bestfit']:.{5 if param == 'r' else 2}f}"
            err_hi, err_lo = errors[0] * scale, errors[1] * scale
            unit = r'\%' if param == 'r' else ''
            if abs(err_hi - err_lo) < 1e-2:
                uncertainty = f'$\\pm{err_hi:.2f}{unit}$'
            else:
                uncertainty = f'$_{{-{err_lo:.2f}{unit}}}^{{+{err_hi:.2f}{unit}}}$'
            return f'{value}{uncertainty} ({confidence}\\%)'

        text_lines = []
        for scan in scans_data:
            text_lines.append(result_line(scan, scan['err_1sig'], 68))
            if sig2:
                text_lines.append(result_line(scan, scan['err_2sig'], 95, False))
        textstr = '\n'.join(text_lines)

        plot.draw([([0.97, 0.955, textstr, 'text'], {
            'transform': ax.transAxes, 'verticalalignment': 'top', 'horizontalalignment': 'right',
            'bbox': {'boxstyle': 'round', 'facecolor': 'white', 'pad': 0.8}, 'fontsize': 20, 'family': 'monospace',
        })])

        # Save outputs
        plot.save(close=False, tight_layout=False)

    finally:
        plot.close()



def plot_2d_scans(
        input_specs: list[tuple[str, ...]],
        outputdir: Path,
        x_param: str = 'Cphi',
        y_param: str = 'Cbox',
        z_cut: int | float = 20.0,
        suffix: str = '',
        right: str = '',
        sig2: bool = False,
        cmap: str = 'plasma',
        coeffs_label: dict[str, str] = {},
        file_formats: list[str] = ['png']
) -> None:
    """
    Plot one or more 2D likelihood scans on the same figure.

    The default axis labels follow the ROOT example used in this analysis:
    Cphi on the x-axis and Cbox on the y-axis.
    """

    from matplotlib.tri import Triangulation

    LOGGER.debug(f'Processing {len(input_specs)} 2D scan(s): {outputdir}')

    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    scans_data = []

    for i, spec in enumerate(input_specs):
        if isinstance(spec, tuple):
            if len(spec) == 3:
                filepath, label, color = spec
            elif len(spec) == 2:
                filepath, label = spec
                color = default_colors[i % len(default_colors)]
            else:
                filepath = spec[0]
                label = f'Scan {i + 1}'
                color = default_colors[i % len(default_colors)]
        else:
            filepath, label = spec, f'Scan {i + 1}'
            color = default_colors[i % len(default_colors)]

        x, y, z, best_x, best_y = process_scan_2d(filepath, x_param, y_param, z_cut)

        scans_data.append({
            'filepath': filepath, 'label': label, 'color': color,
            'x': x, 'y': y, 'z': z, 'best_x': best_x, 'best_y': best_y,
        })

    plot = MatplotlibPlot(outputdir, None, None, 'Scan2D', f'{x_param}_{y_param}{suffix}',
                          file_formats, dpi=300, style=False)
    ax = plot.axes

    try:
        contour_levels = [2.30, 6.18] if sig2 else [2.30]
        # contour_levels = [1.0, 4.0] if sig2 else [1.0]
        contour_handle = None

        for scan in scans_data:
            try:
                tri = Triangulation(scan['x'], scan['y'])
                contourf = plot.draw([([tri, scan['z'], 'tricontourf'], {'levels': 100, 'cmap': cmap, 'alpha': 0.7, 'zorder': 1,})])[0]
                if contour_handle is None: contour_handle = contourf
                plot.draw([([tri, scan['z'], 'tricontour'], {
                    'levels': contour_levels, 'colors': [scan['color']] * len(contour_levels), 'linewidths': 3, 'zorder': 3,
                })])
            except Exception as exc:
                LOGGER.warning(f'Could not build triangulation for {scan["filepath"]}: {exc}')
                plot.draw([([scan['x'], scan['y'], 'scatter'], {'c': scan['z'], 'cmap': cmap, 's': 18, 'alpha': 0.7, 'zorder': 1})])

            plot.draw([([scan['best_x'], scan['best_y'], 'scatter'], {
                's': 180, 'marker': '*', 'color': 'black', 'linewidths': 0.8, 'zorder': 4, 'label': scan['label'],
            })])

        if contour_handle is not None and len(scans_data) == 1:
            cbar = plot.figure.colorbar(contour_handle, ax=ax, pad=0.02)
            cbar.set_label(r'$-2\Delta\ln\Lambda$', loc='top')

        xlabel = coeffs_label.get(x_param, x_param)
        ylabel = coeffs_label.get(y_param, y_param)

        plot.set_labels(xlabel, ylabel, right=right)

        xlim = (min(scan['x'].min() for scan in scans_data), max(scan['x'].max() for scan in scans_data))
        ylim = (min(scan['y'].min() for scan in scans_data), max(scan['y'].max() for scan in scans_data))
        plot.configure(xlim, ylim, legend=True, legend_kwargs={'loc': 'best', 'fontsize': 18})
        ax.grid(True, linestyle=':')

        plot.save(close=False, tight_layout=False)

    finally:
        plot.close()

    return None
