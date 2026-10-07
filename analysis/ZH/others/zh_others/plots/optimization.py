
#################################
### IMPORT STANDARD LIBRARIES ###
#################################

# Standard library and scientific computing imports
import json, uproot

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from package.logger import get_logger

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Import utilities and plotting configurations
from .python.plotter import (
    set_plt_style,
    set_labels,
    savefigs
)



################################
### MATPLOTLIB STYLE SETTING ###
################################

set_plt_style()
plt.ioff()



#######################
### HELPER FUNCTION ###
#######################

def _make_fig(figsize: tuple = (12, 8),
              dpi: int = 100
              ) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=figsize, dpi=dpi)


def _get_chi2_params(chi2_method: str) -> dict[str, any]:
    """Get chi2 method-specific parameters for consistent plot styling.

    Provides method-dependent labels and values for unified visualization
    across different chi2 methods.

    Args:
        chi2_method: Either 'mll' (mass+recoil) or 'pll' (mass+recoil+momentum)

    Returns:
        Dictionary with keys:
        - 'latex_label': X-axis label for plots
        - 'baseline_value': Default parameter value from previous analysis
        - 'baseline_label': Formatted baseline label for legend
    """
    if chi2_method == 'mll':
        return {
            'latex_label': r'$w_{m_{\ell^{+}\ell^{-}}}$',
            'baseline_value': 0.6,
            'baseline_label': r'$w_{m_{\ell^{+}\ell^{-}}}=0.6$'
        }
    elif chi2_method == 'pll':
        return {
            'latex_label': r'$w_{p_{\ell^{+}\ell^{-}}}$',
            'baseline_value': 0.0,
            'baseline_label': r'$w_{p_{\ell^+\ell^-}}=0$'
        }
    else:
        raise ValueError(f"Unknown chi2_method: {chi2_method}")


##############################
### DATA LOADING FUNCTIONS ###
##############################

def load_results(results_file: Path) -> dict[str, dict]:
    """Load optimization scan results from JSON file.

    Parses results produced by optimize.py containing efficiency metrics
    across chi2 parameter space for all event categories.

    Args:
        results_file: Path to the results.json file from optimization

    Returns:
        Dictionary with results keyed by chi2_frac value (as strings)
    """
    if not results_file.exists():
        LOGGER.warning(f'Results file not found: {results_file}\nReturning empty data')
        return {}

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def load_data(root_file: Path) -> dict[str, np.ndarray]:
    """Load kinematic distributions from ROOT file produced by optimize.py.

    Extracts TTree data containing full kinematic information and match quality
    indicators for reconstructed and true particles at specific chi2 values.

    Args:
        root_file: Path to ROOT file (results_baseline.root or results_optimal.root)

    Returns:
        Dictionary mapping variable names to numpy arrays
    """
    if not root_file.exists():
        LOGGER.warning(f'ROOT file not found: {root_file}')
        return {}

    try:
        with uproot.open(root_file) as file:
            tree = file['distributions']
            return {name: np.asarray(tree[name]) for name in tree.keys()}
    except Exception as e:
        LOGGER.error(f'Error loading ROOT file {root_file}:\n{e}')
        return {}


def extract_arrays(results: dict[str, dict], category: str = 'overall') -> tuple[np.ndarray, ...]:
    """Extract numeric arrays from results dictionary for a specific event category.

    Transforms JSON results into numpy arrays sorted by chi2 parameter value,
    ready for plotting and analysis.

    Args:
        results: Results dictionary from JSON file
        category: Event category - 'overall', 'zero_pair', 'one_pair', or 'multi_pair'

    Returns:
        Tuple of sorted arrays: (chi2_fracs, efficiencies, n_correct, n_partial,
                                 n_incorrect, n_total)
    """
    if not results:
        dummy = np.array([])
        return [dummy] * 6

    chi2_fracs =   np.array([float(k) for k in results.keys()])
    efficiencies = np.array([results[k][category]['efficiency']  for k in results.keys()])
    n_correct =    np.array([results[k][category]['n_correct']   for k in results.keys()])
    n_partial =    np.array([results[k][category]['n_partial']   for k in results.keys()])
    n_incorrect =  np.array([results[k][category]['n_incorrect'] for k in results.keys()])
    n_total =      np.array([results[k][category]['n_total']     for k in results.keys()])

    # Sort by chi2_frac
    sort_idx = np.argsort(chi2_fracs)

    return (
        chi2_fracs[sort_idx],
        efficiencies[sort_idx],
        n_correct[sort_idx],
        n_partial[sort_idx],
        n_incorrect[sort_idx],
        n_total[sort_idx]
    )



##########################
### PLOTTING FUNCTIONS ###
##########################

def efficiency(
        chi2_fracs: np.ndarray,
        efficiencies: np.ndarray,
        outDir: Path,
        label: str,
        chi2_method: str,
        full_range: bool = False,
        format: list[str] = ['png']
         ) -> None:
    """Plot pairing efficiency vs chi2 weighting parameter.

    Visualizes how pairing efficiency changes across chi2 parameter space,
    highlighting the optimal parameter value that maximizes efficiency.

    Args:
        chi2_fracs: Array of chi2 parameter values (x-axis)
        efficiencies: Array of corresponding efficiency values (y-axis)
        outDir: Output directory for plots
        label: LaTeX label for the final state (e.g., Z(→μμ)H)
        chi2_method: Either 'mll' or 'pll' for method-specific formatting
        full_range: If True, use full 0-100% y-axis range
    """
    if len(chi2_fracs) == 0:
        return

    # Get method-specific parameters
    params = _get_chi2_params(chi2_method)
    latex_label = params['latex_label']

    # Find optimal chi2 (maximum efficiency)
    optimal_idx  = np.argmax(efficiencies)
    optimal_chi2 = chi2_fracs[optimal_idx]
    optimal_eff  = efficiencies[optimal_idx]

    # Create figure
    fig, ax = _make_fig()
    try:
        # Plot efficiency curve
        ax.scatter(chi2_fracs, efficiencies*100, marker='.')

        # Highlight optimal point
        ax.scatter(optimal_chi2, optimal_eff*100, color='r', marker='*', s=150,
                   label=f'Optimal: {latex_label} = {optimal_chi2:.2f}\nMax Eff.: $\\epsilon = {optimal_eff*100:.2f}\\%$')

        # Add vertical line at optimal point
        ax.axvline(x=optimal_chi2, color='gray', alpha=0.8)

        ax.set_xlim(0, 1)
        if full_range:
            ax.set_ylim(0, 100)

        set_labels(ax, latex_label, r'Pairing Efficiency [\%]', right=label)
        ax.legend()

        savefigs(fig, outDir, 'efficiency', format=format)

    finally:
        plt.close(fig)


def pairing_composition(
        chi2_fracs: np.ndarray,
        n_correct: np.ndarray,
        n_partial: np.ndarray,
        n_incorrect: np.ndarray,
        n_total: np.ndarray,
        efficiencies: np.ndarray,
        outDir: Path,
        label: str,
        chi2_method: str,
        format: list[str] = ['png']
         ) -> None:
    """Plot composition of pairing results (correct, partial, incorrect).

    Args:
        chi2_fracs: Array of chi2 values
        n_correct: Number of correct pairings
        n_partial: Number of partial pairings
        n_incorrect: Number of incorrect pairings
        n_total: Total number of events
        efficiencies: Array of efficiency values
        outDir: Output directory for plots
        label: LaTeX label for the final state
        chi2_method: Either 'mll' or 'pll'
    """
    if len(chi2_fracs) == 0 or np.all(n_total == 0):
        return

    # Get method-specific parameters
    params = _get_chi2_params(chi2_method)
    latex_label = params['latex_label']
    baseline_value = params['baseline_value']

    optimal_idx  = np.argmax(efficiencies)
    optimal_chi2 = chi2_fracs[optimal_idx]
    optimal_eff  = efficiencies[optimal_idx]

    # Get baseline efficiency
    baseline_mask = np.isclose(chi2_fracs, baseline_value, atol=1e-6)
    old_eff = efficiencies[baseline_mask][0] if np.any(baseline_mask) else None

    # Create figure
    fig, ax = _make_fig()

    try:
        # Plot stacked area - use divide with where to avoid warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            correct =   np.divide(n_correct * 100,   n_total, where=n_total > 0, out=np.full_like(n_correct,   np.nan, dtype=float))
            partial =   np.divide(n_partial * 100,   n_total, where=n_total > 0, out=np.full_like(n_partial,   np.nan, dtype=float))
            incorrect = np.divide(n_incorrect * 100, n_total, where=n_total > 0, out=np.full_like(n_incorrect, np.nan, dtype=float))

        ax.fill_between(chi2_fracs, 0, correct,
                        label='Both Correct', color='green', alpha=0.7)
        ax.fill_between(chi2_fracs, correct,
                        correct + partial,
                        label='One Correct', color='orange', alpha=0.7)
        ax.fill_between(chi2_fracs, correct + partial,
                        correct + partial + incorrect,
                        label='Both Incorrect', color='red', alpha=0.7)

        # Optimal chi2 marker
        if optimal_chi2 != baseline_value:
            ax.scatter(optimal_chi2, optimal_eff*100, color='red', marker='*', s=150,
                       label=f'Optimal: {latex_label} = {optimal_chi2:.2f}\n$\\epsilon = {optimal_eff*100:.2f}\\%$')
        # Baseline marker
        if old_eff is not None:
            ax.scatter(baseline_value, old_eff*100, color='green', marker='*', s=150,
                       label=f'Baseline: {latex_label} = {baseline_value}\n$\\epsilon = {old_eff*100:.2f}\\%$')

        # Labels and formatting
        set_labels(ax, latex_label, r'Percentage of Events [\%]', right=label)

        ax.set_xlim(0, 1)
        # Adjust ymin based on minimum efficiency for better readability
        min_efficiency = correct[np.isfinite(correct)].min()
        ymin = 0 if np.isnan(min_efficiency) or min_efficiency < 0.9 else int(min_efficiency - 1)
        ax.set_ylim(ymin, 100)
        ax.legend(loc='lower center')

        # Save figure
        savefigs(fig, outDir, 'pairing_composition', format=format)

    finally:
        plt.close(fig)


def event_counts(
        chi2_fracs: np.ndarray,
        n_correct: np.ndarray,
        n_partial: np.ndarray,
        n_incorrect: np.ndarray,
        outDir: Path,
        label: str,
        chi2_method: str,
        format: list[str] = ['png']
         ) -> None:
    """Plot absolute event counts for different pairing outcomes.

    Args:
        chi2_fracs: Array of chi2 values
        n_correct: Number of correct pairings
        n_partial: Number of partial pairings
        n_incorrect: Number of incorrect pairings
        outDir: Output directory for plots
        label: LaTeX label for the final state
        chi2_method: Either 'mll' or 'pll'
    """
    if len(chi2_fracs) == 0:
        return

    # Get method-specific parameters
    params = _get_chi2_params(chi2_method)
    latex_label = params['latex_label']

    # Create figure
    fig, ax = _make_fig()

    try:
        # Plot lines for each category
        ax.scatter(chi2_fracs, n_correct,   color='green',  marker='.', label='Both Correct')
        ax.scatter(chi2_fracs, n_partial,   color='orange', marker='.', label='One Correct')
        ax.scatter(chi2_fracs, n_incorrect, color='red',    marker='.', label='Both Incorrect')

        # Labels and formatting
        set_labels(ax, latex_label, 'Number of Events', right=label)
        ax.set_yscale('log')
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, None)
        ax.legend()

        # Save figure
        savefigs(fig, outDir, 'event_counts', format=format)

    finally:
        plt.close(fig)


def plot_comp(
        old: np.ndarray,
        reco: np.ndarray,
        true: np.ndarray,
        var_name: str,
        xlabel: str,
        outDir: Path,
        label: str,
        chi2_method: str,
        bins: int = 100,
        log_scale: str = 'linear',
        format: list[str] = ['png']
         ) -> None:
    """Plot kinematic variable distributions comparing baseline and optimal chi2 values.

    Creates overlaid histograms showing reconstructed and true kinematics at
    baseline and optimal chi2 parameters to visualize improvement.

    Args:
        old: Reconstructed values at baseline chi2
        reco: Reconstructed values at optimal chi2
        true: True MC values (for reference)
        var_name: Variable identifier for filename
        xlabel: LaTeX label for x-axis
        outDir: Output directory for plots
        label: LaTeX label for final state
        chi2_method: 'mll' or 'pll' for method-specific formatting
        bins: Number of histogram bins
        log_scale: 'linear' or 'log' for y-axis scale
    """
    # Check if arrays are not empty
    if len(old) == 0 or len(reco) == 0 or len(true) == 0:
        LOGGER.warning(f'Empty array for {var_name}, skipping plot')
        return

    # Get method-specific parameters
    params = _get_chi2_params(chi2_method)
    baseline_label = params['baseline_label']

    # Create figure
    fig, ax = _make_fig()

    try:
        # Determine common range for all histograms
        xmin = min(old.min(), reco.min(), true.min())
        xmax = max(old.max(), reco.max(), true.max())

        # Plot histograms
        ax.hist(old, bins=bins, range=(xmin, xmax), histtype='step',
                label=baseline_label, color='blue', density=True)
        ax.hist(reco, bins=bins, range=(xmin, xmax), histtype='step',
                label=r'Optimal', color='red', density=True)
        ax.hist(true, bins=bins, range=(xmin, xmax),
                label='True', color='green', density=True)

        set_labels(ax, xlabel, 'Unit Area', right=label)

        ax.set_xlim(xmin, xmax)
        if log_scale == 'log':
            ax.set_yscale('log')
        ax.legend()

        # Save figure
        savefigs(fig, outDir, var_name, '_'+log_scale, format)

    finally:
        plt.close(fig)


def plot_origin(
        old: np.ndarray,
        reco: np.ndarray,
        true: np.ndarray,
        matches: np.ndarray,
        var_name: str,
        xlabel: str,
        outDir: Path,
        label: str,
        chi2_method: str,
        bins: int = 100,
        log_scale: str = 'linear',
        format: list[str] = ['png']
         ) -> None:
    """Plot kinematic variables colored by pairing match quality.

    Creates overlaid histograms showing reconstructed kinematics at optimal chi2,
    color-coded by match quality (correct, partial, incorrect) compared to MC truth.
    Baseline values shown as scatter points for reference.

    Args:
        old: Reconstructed values at baseline chi2
        reco: Reconstructed values at optimal chi2 (main data)
        true: True MC values (reference)
        matches: Match quality array (0=incorrect, 1=partial, 2=correct)
        var_name: Variable identifier for filename
        xlabel: LaTeX label for x-axis
        outDir: Output directory for plots
        label: LaTeX label for final state
        chi2_method: 'mll' or 'pll' for method-specific formatting
        bins: Number of histogram bins
        log_scale: 'linear' or 'log' for y-axis scale
    """
    # Check if arrays are not empty
    if len(old) == 0 or len(reco) == 0 or len(true) == 0:
        LOGGER.warning(f'Empty array for {var_name}, skipping plot')
        return

    # Get method-specific parameters
    params = _get_chi2_params(chi2_method)
    baseline_label = params['baseline_label']

    # Create figure
    fig, ax = _make_fig()

    try:
        # Determine common range for all histograms
        xmin = min(old.min(), reco.min(), true.min())
        xmax = max(old.max(), reco.max(), true.max())

        for m, leg, c in [(0, 'Both Incorrect', 'red'), (1, 'One Correct', 'orange'), (2, 'Both Correct', 'green')]:
            matche = reco[matches == m]
            ax.hist(matche, bins=bins, range=(xmin, xmax), histtype='step',
                    label=leg, color=c, linewidth=2)

        # Plot histograms
        n1, b = np.histogram(old,  bins=bins, range=(xmin, xmax))
        n2, _ = np.histogram(reco, bins=bins, range=(xmin, xmax))
        B = (b[1:] + b[:-1]) / 2
        ax.scatter(B, n1, marker='.', color='tab:blue',   label=baseline_label)
        ax.scatter(B, n2, marker='.', color='tab:orange', label=r'Optimal')

        set_labels(ax, xlabel, 'Events', right=label)

        ax.set_xlim(xmin, xmax)
        if log_scale == 'log':
            ax.set_yscale('log')
        ax.legend()

        # Save figure
        savefigs(fig, outDir, var_name, '_'+log_scale, format)

    finally:
        plt.close(fig)


def compare_dists(
        old_file: Path,
        optimal_file: Path,
        out_comp: Path,
        out_origin: Path,
        label: str,
        chi2_method: str,
        format: list[str] = ['png']
         ) -> None:
    """Compare kinematic distributions between baseline and optimal chi2 values.

    Loads ROOT files containing full kinematic information and generates
    comparison plots for leptons, Z system, and pair variables, color-coded
    by pairing match quality.

    Args:
        old_file: Path to results_baseline.root
        optimal_file: Path to results_optimal.root
        out_comp: Output directory for direct comparison plots
        out_origin: Output directory for plots colored by match quality
        label: LaTeX label for the final state
        chi2_method: Either 'mll' or 'pll'
    """
    LOGGER.info('Loading distribution files')
    old_dists = load_data(old_file)
    opt_dists = load_data(optimal_file)

    if not old_dists or not opt_dists:
        LOGGER.warning('Could not load distributions, skipping plot')
        return

    LOGGER.info('Generating distribution comparison plots')

    # Define variable pairs (reconstructed, true)
    comparisons = [
        # Leptons
        ('leading_p',     'Leading_p',     r'$p_{leading}$'),
        ('leading_pt',    'Leading_pt',    r'$p_{T,leading}$'),
        ('leading_theta', 'Leading_theta', r'$\theta_{leading}$'),

        ('subleading_p',     'Subleading_p',     r'$p_{subleading}$'),
        ('subleading_pt',    'Subleading_pt',    r'$p_{T,subleading}$'),
        ('subleading_theta', 'Subleading_theta', r'$\theta_{subleading}$'),

        # Z system
        ('zll_p',     'Zll_p',     r'$p_{\ell^{+}\ell^{-}}$'),
        ('zll_pt',    'Zll_pt',    r'$p_{T,\ell^{+}\ell^{-}}$'),
        ('zll_theta', 'Zll_theta', r'$\theta_{\ell^{+}\ell^{-}}$'),

        # Pair variables
        ('mass',   'Mass',   r'$m_{\ell^{+}\ell^{-}}$'),
        ('recoil', 'Recoil', r'$m_{recoil}$'),
    ]

    # Create output directory
    out_comp.mkdir(parents=True, exist_ok=True)
    out_origin.mkdir(parents=True, exist_ok=True)

    # Generate comparison plots for each distribution
    for reco_var, true_var, display_name in comparisons:
        # Check if variables exist in both files
        if reco_var not in old_dists or true_var not in old_dists:
            LOGGER.warning(f'Variables {reco_var}/{true_var} not found in baseline file')
            continue

        if reco_var not in opt_dists or true_var not in opt_dists:
            LOGGER.warning(f'Variables {reco_var}/{true_var} not found in optimal file')
            continue

        # Plot for both linear and log scales
        for log_scale in ['linear', 'log']:
            plot_comp(
                old_dists[reco_var],
                opt_dists[reco_var],
                opt_dists[true_var],
                reco_var,
                display_name,
                out_comp,
                label,
                chi2_method,
                log_scale=log_scale,
                format=format
            )
            plot_origin(
                old_dists[reco_var],
                opt_dists[reco_var],
                opt_dists[true_var],
                opt_dists['matches'],
                reco_var,
                display_name,
                out_origin,
                label,
                chi2_method,
                log_scale=log_scale,
                format=format
            )
