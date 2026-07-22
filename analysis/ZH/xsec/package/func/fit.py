####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import os, json, uproot, subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from pathlib import Path
from scipy.interpolate import UnivariateSpline
from uuid import uuid4
from datetime import datetime

from ..logger import get_logger

LOGGER = get_logger(__name__)




###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from package.userConfig import plot_file, PathObj
from package.plots.python.plotter import (
    set_plt_style,
    set_labels,
    savefigs
)
from package.func.self_coupling import (
    kappa_from_SMEFT,
    kappa_precision
)


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

coeffs_label = {
    'r':     r'$\mu_{ZH}$',
    'Cphi':  r'$C_{\phi}$',
    'CphiD': r'$C_{\phi D}$',
    'Cbox':  r'$C_{\mathrm{Box}}$',
}



########################
### HELPER FUNCTIONS ###
########################

def add_stamp(
        path: PathObj,
        label: str,
        status: str = 'ok'
         ) -> None:
    '''Add a timestamped stamp to a log file for tracking execution status.'''

    # Generate timestamp and unique ID
    ts = datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
    uniq = uuid4().hex[:8]
    stamp = f'\n\n---- STAMP {label} [{status}]: {ts} | id={uniq}\n'

    # Append stamp to log file
    with open(path, 'a') as log_file:
        log_file.write(stamp)
    LOGGER.debug(f'Added STAMP: {ts} | id={uniq} | status={status} | file={label}')

def mk_csv(
    params: list[str],
    limits: dict[float | int],
    ngrids: dict[str, int],
    eps: float = 1e-4,
) -> pd.DataFrame:

    from itertools import product

    def _log_segment(lo: float, hi: float, n: int) -> np.ndarray:
        lo = max(lo, eps)
        hi = max(hi, eps)
        if n <= 0:
            return np.empty(0)
        if n == 1:
            return np.array([hi])
        return np.logspace(np.log10(lo), np.log10(hi), n)

    grids: dict[str, np.ndarray] = {}

    for param in params:
        limit, ngrid = limits[param], ngrids[param]
        Dw, Up = float(limit[0]), float(limit[1])
        nside = ngrid // 2 if ngrid % 2 == 0 else (ngrid - 1) // 2

        if Dw < 0 < Up:
            left = -_log_segment(eps, abs(Dw), nside)[::-1]
            right = _log_segment(eps, abs(Up), nside)
            tab = np.concatenate([left, np.array([0.0]), right])

        else:
            center = 0.5 * (Dw + Up)
            dlow = abs(Dw - center)
            dhigh = abs(Up - center)

            left = center - _log_segment(eps, dlow, nside)[::-1]
            right = center + _log_segment(eps, dhigh, nside)
            tab = np.concatenate([left, np.array([center]), right])

        grids[param] = tab

    df = pd.DataFrame(list(product(*(grids[param] for param in params))), columns=params)

    return df


def run_cmd(cmd: list[str] | str,
            log_file: Path | PathObj | str | None = None,
            cwd: Path | PathObj | str | None = None,
            env: os._Environ | None = None
            ) -> int:
    '''Run a command using subprocess'''
    if isinstance(log_file, str): log_file = PathObj(log_file)
    status = 'not-run'
    try:
        if log_file is not None:
            with open(log_file, 'w') as out:
                result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=cwd, env=env, check=True)
        else:
            result = subprocess.run(cmd, text=True, capture_output=False, cwd=cwd, env=env, check=True)
        status = 'ok'
    except subprocess.CalledProcessError as exc:
        if status == 'not-run': status = f'error exit = {exc.returncode}'
        LOGGER.error(f'Command failed with exit code {exc.returncode}')
        return exc.returncode
    finally:
        if log_file is not None and log_file.exists():
            add_stamp(log_file, log_file.name, status)

    return result.returncode


def get_grid_number(n: int, params: list[str] | int, pm: bool = True) -> int:
    if n % 2 == 0:
        n = n+1 if pm else n-1
    if isinstance(params, int):
        return n**params
    return n**len(params)



##########################
### RESULTS EXTRACTION ###
##########################

def check_log(res_log: str) -> int | None:
    '''Check if the fit has converged'''
    LOGGER.debug('Fit done, checking fit log')

    # Parse log file for signal strength result
    # (parse from end to find latest result)
    status = None, False
    with open(str(res_log)) as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'WARNING: MultiDimFit failed' in line:
                status = -1
                break
        if status is None:
            for line in reversed(lines):
                if 'Minimization finished with status=' in line:
                    status = int(line.split('=')[-1])

    if status is None:
        LOGGER.error(f"Couldn't find minimization status at {res_log}")
    elif status == 0:
        LOGGER.debug(f'Minimization success, {status = }')
    elif status == 1:
        LOGGER.warning(f'Minimization finished with {status = }\n'
                       f'Check the log at {res_log}')
    elif status == -1:
        LOGGER.error('Minimization did not converge\n'
                     f'Check the log at {res_log}')
    return status


def get_results(file, params, algo) -> dict[str, float | int]:

    res = {}
    if isinstance(params, str): params = [params]
    tree: dict[str, np.ndarray] = uproot.open(file)['limit'].arrays(params, library='np')

    if algo == 'singles':
        for i, param in enumerate(params):
            bestfit = float(tree[param][0])
            if bestfit < 0 and bestfit > -1e-2: bestfit = abs(bestfit)
            err_lo, err_hi = abs(float(tree[param][2 * i + 1])), abs(float(tree[param][2 * i + 2]))
            res[param] = {'best_fit': bestfit, 'err_lo': err_lo, 'err_hi': err_hi, 'err': (err_hi + err_lo) / 2}

    elif algo == 'grid':
        for param in params:
            other_params = [p for p in params if p!=param] if len(params)>1 else []
            bestfit, err_hi, err_lo = process_scan(file, param, 1e3, True, other_params)
            res[param] = {'best_fit': bestfit, 'err_lo': err_lo, 'err_hi': err_hi, 'err': (err_hi + err_lo) / 2}
    else:
        raise ValueError(f'{algo = } is not supported, choose between [singles, grid]')

    return res


def convert_to_kappa(
        res: dict[str, dict[str, float | int]],
        lbda: float | int = 1e3,
         ) -> tuple[float | int, float | int]:
    cphi, cphid, cbox = res.get('Cphi', {}), res.get('CphiD', {}),  res.get('Cbox', {})

    cphi_val,  cphi_err  = cphi.get('best_fit',  0), cphi.get('err',  0)
    cphid_val, cphid_err = cphid.get('best_fit', 0), cphid.get('err', 0)
    cbox_val,  cbox_err  = cbox.get('best_fit',  0), cbox.get('err',  0)

    kappa_val = kappa_from_SMEFT(cphi_val, cphid_val, cbox_val, lbda)
    kappa_err = kappa_precision(cphi_err,  cphid_err, cbox_err, lbda)

    LOGGER.info('Converting to kappa framework:\n  '
                f'kappa_lbda = {kappa_val:.2f} +/- {kappa_err*100:.2f}%')

    return kappa_val, kappa_err




######################
### RESULTS SAVING ###
######################

def res_saving(
        results: dict[str, float | int],
        outDir: str,
        print_result: bool = True,
        target: str = '',
        eps: float = 1e-2
         ) -> None:
    '''Save fitting results (signal strength and uncertainty) to output file.'''
    if not results:
        LOGGER.warning("Couldn't extract values of the fit, go to the log file to have more information")
        return

    LOGGER.info('Results successfully extracted')
    for param, res in results.items():
        bestfit, err   = res['best_fit'], res['err']
        err_hi, err_lo = res['err_hi'],   res['err_lo']
        precision = 6 if param == 'r' else 2
        symmetric = abs(err_hi - err_lo) < eps
        result_text = (
            f'  {param} = {bestfit:.{precision}f} +/- {err:.{precision}f}'
            if symmetric else
            f'  {param} = {bestfit:.{precision}f} +{err_hi:.{precision}f}/-{err_lo:.{precision}f}'
        )
        if print_result:
            LOGGER.info(result_text)
            if param == 'r':
                unc_text = f'{err * 100:.2f} %' if symmetric else f'+{err_hi * 100:.2f}/-{err_lo * 100:.2f} %'
                LOGGER.info(f'Uncertainty obtained on ZH cross-section: {unc_text}')

    out_file = Path(outDir) / f'results{target}.json'
    out_file.write_text(json.dumps(results))

    LOGGER.debug(f'Saved results in {out_file}')



##########################
### PLOTTING FUNCTIONS ###
##########################

def read_tree_data(
        file_path: Path,
        param: str = "r",
        y_cut: float | int = 7.0,
        other_params: list[str] | None = None,
        eps: float = 1e-5
         ) -> tuple[np.ndarray,
                    np.ndarray]:
    """
    Read limit tree from ROOT file using uproot.
    Returns sorted parameter and 2*deltaNLL arrays.
    """
    with uproot.open(file_path) as f:
        branches = [param, "deltaNLL"]
        if other_params:
            branches.extend(other_params)

        tree = f["limit"].arrays(branches, library='np')

        if other_params:
            # Keep all points where the auxiliary parameters are within the
            # requested tolerance around zero.
            mask = np.logical_and.reduce([
                np.abs(tree[p]) <= eps for p in other_params
            ])
            tree = {key: values[mask] for key, values in tree.items()}

        # Read data directly as numpy arrays
        x_data = tree[param]
        y_data = 2 * tree["deltaNLL"]

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


def read_tree_data_2d(
        file_path: Path,
        x_param: str = 'Cphi',
        y_param: str = 'Cbox',
        z_cut: float | int = 1e0
         ) -> tuple[np.ndarray,
                    np.ndarray,
                    np.ndarray]:
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

        raw_data: pd.DataFrame = tree.arrays([x_param, y_param, 'deltaNLL'], library='pd')
        raw_data['deltaNLL'] *= 2

    LOGGER.debug(f'Found {raw_data.shape[0]} point in {file_path}')

    data = raw_data[raw_data['deltaNLL']<=z_cut].to_numpy()
    LOGGER.debug(f'Removed points greater than {z_cut = }, {len(data)} points remaining')

    if len(data) == 0:
        return data[:, 0], data[:, 1], data[:, 2]

    coords = data[:, :2]
    uniq_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    uniq_z = np.full(len(uniq_coords), np.inf)
    np.minimum.at(uniq_z, inverse, data[:, 2])

    data = np.column_stack([uniq_coords, uniq_z])
    data = data[data[:, 0].argsort(kind='mergesort')]

    LOGGER.debug(f'Removed {len(raw_data) - len(data)} duplicate, {len(data)} points remaining')

    return data[:, 0], data[:, 1], data[:, 2]


def process_scan_2d(
        input_file: Path,
        x_param: str = 'Cphi',
        y_param: str = 'Cbox',
        z_cut: float | int = 50.0
         ) -> tuple[np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    float,
                    float]:
    """
    Process a single 2D likelihood scan and return the scan points and best-fit coordinates.
    """
    x, y, z = read_tree_data_2d(input_file, x_param, y_param, z_cut)

    if len(x) <= 2:
        raise ValueError(f"Not enough points in {input_file}: {len(x)}")

    best_idx = z == 0
    best_x, best_y = x[best_idx], y[best_idx]

    return x, y, z, best_x, best_y


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
        y_cut: float | int = 7.0,
        only_res: bool = False,
        other_params: list[str] = [],
        eps: float = 1e-5
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
    x, y = read_tree_data(input_file, param, y_cut, other_params, eps)

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

    if only_res:
        return float(bestfit), float(err_hi1), float(err_lo1)
    return x, y, bestfit, spl, (err_hi1, err_lo1), (err_hi2, err_lo2)


def plot_1d_scans(
        input_specs: list[tuple[str, ...]],
        output: Path,
        param: str = 'r',
        y_cut: int | float = 7.0,
        y_max: int | float = -1,
        suffix: str = '',
        right: str = '',
        sig2: bool = False,
        other_params: list[str] = []
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

        x, y, bestfit, spl, err_1sig, err_2sig = process_scan(filepath, param, y_cut, other_params=other_params)
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = spl(x_smooth)
        if (y_max > 0) and (y_max < y_cut):
            x_smooth = x_smooth[y_smooth<=y_max]
            y_smooth = y_smooth[y_smooth<=y_max]
            x = x[y<=y_max]
            y = y[y<=y_max]
        else:
            x_smooth = x_smooth[y_smooth<=y_cut]
            y_smooth = y_smooth[y_smooth<=y_cut]
            x = x[y<=y_cut]
            y = y[y<=y_cut]


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
            if param == 'r':
                ax.scatter((scan['x']-1)*100, scan['y'], s=15, color=scan['color'], zorder=2)
                ax.plot((scan['x_smooth']-1)*100, scan['y_smooth'], '-', linewidth=2.5,
                        color=scan['color'], label=scan['label'], zorder=3)
            else:
                ax.scatter(scan['x'], scan['y'], s=15, color=scan['color'], zorder=2)
                ax.plot(scan['x_smooth'], scan['y_smooth'], '-', linewidth=2.5,
                        color=scan['color'], label=scan['label'], zorder=3)
            for yv in [1.0, 4.0]:
                crossings_yv = find_crossings(scan['x'], scan['spl'], yv)
                for cr in crossings_yv:
                    if param == 'r':
                        ax.plot([(cr['lo']-1)*100, (cr['lo']-1)*100], [0, yv], color=scan['color'], alpha=0.5, linestyle='dashed')
                        ax.plot([(cr['hi']-1)*100, (cr['hi']-1)*100], [0, yv], color=scan['color'], alpha=0.5, linestyle='dashed')
                    else:
                        ax.plot([cr['lo'], cr['lo']], [0, yv], color=scan['color'], alpha=0.5, linestyle='dashed')
                        ax.plot([cr['hi'], cr['hi']], [0, yv], color=scan['color'], alpha=0.5, linestyle='dashed')

        # Draw horizontal reference lines for 1σ and 2σ
        ax.axhline(1.0,   color='gray', linestyle='dashed')
        ax.axhline(4.0,   color='gray', linestyle='dashed')
        if (y_max > 0) and (y_max < y_cut):
            ax.axhline(y_max, color='black')
        else:
            ax.axhline(y_cut, color='black')


        # Formatting
        lab = coeffs_label.get(param, param)
        if param == 'r':
            set_labels(ax, r'$\mu_{ZH}-1$ [\%]', r'$-2\Delta\ln\Lambda$', right=right)
        else:
            set_labels(ax, lab, r'$-2\Delta\ln\Lambda$', right=right)

        # Calculate required y_max based on data and text
        if (y_max > 0):
            max_y_smooth = max([np.max(scan['y_smooth']) for scan in scans_data] + [min([y_max, y_cut])])
        else:
            max_y_smooth = max([np.max(scan['y_smooth']) for scan in scans_data] + [y_max, y_cut])
        num_text_lines = sum(1 + (1 if sig2 else 0) for _ in scans_data)
        text_box_height = num_text_lines * 1.1
        y_max_required = max_y_smooth + text_box_height + 0.5   # + 0.5 margin
        effective_y_max = max(y_max, y_max_required)

        ax.set_ylim(0, effective_y_max)

        # Set x-axis limits based on all scans
        all_x = np.concatenate([s['x_smooth'] for s in scans_data])
        margin = 0.05 * (all_x.max() - all_x.min())

        if param == 'r':
            ax.set_xlim((all_x.min() - margin - 1)*100, (all_x.max() + margin - 1)*100)
        else:
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)

        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=22)

        # Add text box with fit results (all scans)
        text_lines = []
        for scan in scans_data:
            err_hi_1, err_lo_1 = scan['err_1sig']
            err_hi_2, err_lo_2 = scan['err_2sig']

            if param == 'r':
                if abs(err_hi_1 - err_lo_1) * 100 < 1e-2:
                    line_68 = rf"{scan['label']}: {lab} = {scan['bestfit']:.5f}$\pm{err_hi_1*100:.2f}\%$ (68\%)"
                else:
                    line_68 = f"{scan['label']}: {lab} = {scan['bestfit']:.5f}$_{{-{err_lo_1*100:.2f}\\%}}^{{+{err_hi_1*100:.2f}\\%}}$ (68\\%)"
                text_lines.append(line_68)

                if sig2:
                    prefix = ' ' * len(scan['label'])
                    if abs(err_hi_2 - err_lo_2) * 100 < 1e-2:
                        line_95 = rf"{prefix}: {lab} = {scan['bestfit']:.5f}$\pm{err_hi_2*100:.2f}\%$ (95\%)"
                    else:
                        line_95 = f"{prefix}: {lab} = {scan['bestfit']:.5f}$_{{-{err_lo_2*100:.2f}\\%}}^{{+{err_hi_2*100:.2f}\\%}}$ (95\\%)"
                    text_lines.append(line_95)
            else:
                if abs(err_hi_1 - err_lo_1) < 1e-2:
                    line_68 = rf"{scan['label']}: {lab} = {scan['bestfit']:.2f}$\pm{err_hi_1:.2f}$ (68\%)"
                else:
                    line_68 = f"{scan['label']}: {lab} = {scan['bestfit']:.2f}$_{{-{err_lo_1:.2f}}}^{{+{err_hi_1:.2f}}}$ (68\\%)"
                text_lines.append(line_68)

                if sig2:
                    prefix = ' ' * len(scan['label'])
                    if abs(err_hi_2 - err_lo_2) < 1e-2:
                        line_95 = rf"{prefix}: {lab} = {scan['bestfit']:.2f}$\pm{err_hi_2:.2f}$ (95\%)"
                    else:
                        line_95 = f"{prefix}: {lab} = {scan['bestfit']:.2f}$_{{-{err_lo_2:.2f}}}^{{+{err_hi_2:.2f}}}$ (95\\%)"
                    text_lines.append(line_95)

        textstr = "\n".join(text_lines)

        props = dict(boxstyle='round', facecolor='white', pad=0.8)
        ax.text(0.97, 0.955, textstr, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, fontsize=20, family='monospace')

        # Save outputs
        savefigs(fig, output, 'Scan', suffix, plot_file)

    finally:
        plt.close(fig)



def plot_2d_scans(
        input_specs: list[tuple[str, ...]],
        output: Path,
        x_param: str = 'Cphi',
        y_param: str = 'Cbox',
        z_cut: int | float = 20.0,
        suffix: str = '',
        right: str = '',
        sig2: bool = False,
        cmap: str = 'plasma'
         ) -> None:
    """
    Plot one or more 2D likelihood scans on the same figure.

    The default axis labels follow the ROOT example used in this analysis:
    Cphi on the x-axis and Cbox on the y-axis.
    """
    LOGGER.info(f'Processing {len(input_specs)} 2D scan(s): {output}')

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
            'filepath': filepath,
            'label': params_label.get(label, label),
            'color': color,
            'x': x,
            'y': y,
            'z': z,
            'best_x': best_x,
            'best_y': best_y,
        })

    fig, ax = plt.subplots(dpi=300)

    try:
        contour_levels = [2.30, 6.18] if sig2 else [2.30]
        # contour_levels = [1.0, 4.0] if sig2 else [1.0]
        contour_handle = None

        for scan in scans_data:
            try:
                tri = Triangulation(scan['x'], scan['y'])
                contourf = ax.tricontourf(tri, scan['z'], levels=100, cmap=cmap, alpha=0.7,
                                          zorder=1)
                if contour_handle is None:
                    contour_handle = contourf
                ax.tricontour(tri, scan['z'], levels=contour_levels,
                              colors=[scan['color']] * len(contour_levels),
                              linewidths=3, zorder=3)
            except Exception as exc:
                LOGGER.warning(f'Could not build triangulation for {scan["filepath"]}: {exc}')
                ax.scatter(scan['x'], scan['y'], c=scan['z'], cmap=cmap,
                           s=18, alpha=0.7, zorder=1)

            ax.scatter(scan['best_x'], scan['best_y'], s=180, marker='*',
                       color='black', linewidths=0.8,
                       zorder=4, label=scan['label'])

        if contour_handle is not None and len(scans_data) == 1:
            cbar = fig.colorbar(contour_handle, ax=ax, pad=0.02)
            cbar.set_label(r'$-2\Delta\ln\Lambda$', loc='top')

        xlabel = coeffs_label.get(x_param, x_param)
        ylabel = coeffs_label.get(y_param, y_param)

        set_labels(ax, xlabel, ylabel, right=right)

        all_x = np.concatenate([scan['x'] for scan in scans_data])
        all_y = np.concatenate([scan['y'] for scan in scans_data])
        if len(all_x) > 1:
            ax.set_xlim(all_x.min(), all_x.max())
        if len(all_y) > 1:
            ax.set_ylim(all_y.min(), all_y.max())

        ax.grid(True, linestyle=':')
        ax.legend(loc='best', fontsize=18)

        suffix = f'{x_param}_{y_param}{suffix}'
        savefigs(fig, output, 'Scan2D', suffix, plot_file)

    finally:
        plt.close(fig)

    return None
