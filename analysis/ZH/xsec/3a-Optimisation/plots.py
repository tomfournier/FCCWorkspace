#!/usr/bin/env python3
"""
Plotting script for chi2_recoil_frac optimization results.

This script reads the JSON results from optimize_ll.py and generates
comprehensive plots to visualize the pairing efficiency as a function
of the chi2_recoil_frac parameter.

Usage:
    python3 plots.py --cat mumu --ecm 240 [--procs process1-process2]
"""

##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

print('----->[Info] Loading modules')

# Standard library and scientific computing imports
import sys, json, uproot

from time import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Start execution timer
t = time()

print('----->[Info] Loading custom modules')

# Import configuration paths and plot settings
from package.userConfig import loc, plot_file
# Import utilities and plotting configurations
from package.config import timer, process_label
from package.plots.python.plotter import (
    set_plt_style,
    set_labels,
    savefigs
)


########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args
parser = create_parser(
    cat_single=True,
    optimize=True,
    is_plot=True,
    description='Optimisation Plots Script'
)
arg = parse_args(parser, validate_cat=True)


set_plt_style()
plt.ioff()



#######################
### HELPER FUNCTION ###
#######################

def _make_fig(figsize: tuple = (12, 8),
              dpi: int = 100
              ) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=figsize, dpi=dpi)


##########################
### PLOTTING FUNCTIONS ###
##########################

def load_results(results_file: Path) -> dict[str, dict]:
    """Load optimization results from JSON file.

    Args:
        results_file: Path to the results.json file

    Returns:
        Dictionary with results keyed by chi2_frac
    """
    if not results_file.exists():
        print(f"Warning: Results file not found: {results_file}")
        return {}

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def extract_arrays(results: dict[str, dict], category: str = 'overall') -> tuple[np.ndarray, ...]:
    """Extract numpy arrays from results dictionary for a specific category.

    Args:
        results: Results dictionary from JSON
        category: Category key ('overall', 'zero_pair', 'one_pair', 'multi_pair')

    Returns:
        Tuple of (chi2_fracs, efficiencies, n_correct, n_partial, n_incorrect, n_total)
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


def efficiency(
        chi2_fracs: np.ndarray,
        efficiencies: np.ndarray,
        outDir: Path,
        label: str,
        full_range: bool = False) -> None:
    """Plot efficiency vs chi2_recoil_frac.

    Args:
        chi2_fracs: Array of chi2_recoil_frac values
        efficiencies: Array of efficiency values
        proc_name: Process name for title
        cat: Final state category (ee or mumu)
        outDir: Output directory for plots
    """
    if len(chi2_fracs) == 0:
        return

    # Find optimal chi2_frac (maximum efficiency)
    optimal_idx = np.argmax(efficiencies)
    optimal_chi2 = chi2_fracs[optimal_idx]
    optimal_eff = efficiencies[optimal_idx]

    # Create figure
    fig, ax = _make_fig()
    try:
        # Plot efficiency curve
        ax.scatter(chi2_fracs, efficiencies*100, marker='.')

        # Highlight optimal point
        ax.scatter(optimal_chi2, optimal_eff*100, color='r', marker='*', s=150)

        # Add vertical line at optimal point
        ax.axvline(x=optimal_chi2, color='gray', alpha=0.8,
                   label=(f'Optimal: $w_{{\\ell^{{+}}\\ell^{{-}}}}$ = {optimal_chi2:.2f}\n'
                          f'Max Eff.: $\\epsilon = {optimal_eff*100:.2f}\\%$'))

        ax.set_xlim(0, 1)
        if full_range: ax.set_ylim(0, 1)

        set_labels(ax, r'$w_{\ell^{+}\ell^{-}}$',
                   r'Pairing Efficiency [\%]', right=label)
        ax.legend()

        savefigs(fig, outDir, 'efficiency', format=plot_file)

    finally:
        plt.close(fig)


def pairing_composition(
        chi2_fracs: np.ndarray,
        n_correct: np.ndarray,
        n_partial: np.ndarray,
        n_incorrect: np.ndarray,
        n_total: np.ndarray,
        outDir: Path,
        label: str,
         ) -> None:
    """Plot composition of pairing results (correct, partial, incorrect).

    Args:
        chi2_fracs: Array of chi2_recoil_frac values
        n_correct: Number of correct pairings
        n_partial: Number of partial pairings
        n_incorrect: Number of incorrect pairings
        n_total: Total number of events
        proc_name: Process name for title
        cat: Final state category (ee or mumu)
        outDir: Output directory for plots
    """
    if len(chi2_fracs) == 0 or np.all(n_total == 0):
        return

    # Create figure
    fig, ax = _make_fig()

    try:
        # Plot stacked area - use divide with where to avoid warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            correct   = np.divide(n_correct,   n_total, where=n_total!=0) * 100
            partial   = np.divide(n_partial,   n_total, where=n_total!=0) * 100
            incorrect = np.divide(n_incorrect, n_total, where=n_total!=0) * 100

        ax.fill_between(chi2_fracs, 0, correct,
                        label='Both Correct', color='green', alpha=0.7)
        ax.fill_between(chi2_fracs, correct,
                        correct + partial,
                        label='One Correct', color='orange', alpha=0.7)
        ax.fill_between(chi2_fracs, correct + partial,
                        correct + partial + incorrect,
                        label='Both Incorrect', color='red', alpha=0.7)

        # Labels and formatting
        set_labels(ax, r'$w_{\ell^{+}\ell^{-}}$', r'Percentage of Events [\%]', right=label)

        ax.set_xlim(0, 1)
        # Adjust ymin based on minimum efficiency for better readability
        min_efficiency = correct[np.isfinite(correct)].min()
        ymin = 0 if np.isnan(min_efficiency) or min_efficiency < 0.9 else int(min_efficiency - 1)
        ax.set_ylim(ymin, 100)
        ax.legend(loc='lower center')

        # Save figure
        savefigs(fig, outDir, 'pairing_composition', format=plot_file)

    finally:
        plt.close(fig)


def event_counts(
        chi2_fracs: np.ndarray,
        n_correct: np.ndarray,
        n_partial: np.ndarray,
        n_incorrect: np.ndarray,
        outDir: Path,
        label: str) -> None:
    """Plot absolute event counts for different pairing outcomes.

    Args:
        chi2_fracs: Array of chi2_recoil_frac values
        n_correct: Number of correct pairings
        n_partial: Number of partial pairings
        n_incorrect: Number of incorrect pairings
        proc_name: Process name for title
        cat: Final state category (ee or mumu)
        outDir: Output directory for plots
    """
    if len(chi2_fracs) == 0:
        return

    # Create figure
    fig, ax = _make_fig()

    try:
        # Plot lines for each category
        ax.scatter(chi2_fracs, n_correct,   color='green',  marker='.', label='Both Correct')
        ax.scatter(chi2_fracs, n_partial,   color='orange', marker='.', label='One Correct')
        ax.scatter(chi2_fracs, n_incorrect, color='red',    marker='.', label='Both Incorrect')
        # ax.scatter(chi2_fracs, n_total,     color='blue',   marker='.', label='Total events', alpha=0.5)

        # Labels and formatting
        set_labels(ax, r'$w_{\ell^{+}\ell^{-}}$', 'Number of Events', right=label)
        ax.set_yscale('log')
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, None)
        ax.legend()

        # Save figure
        savefigs(fig, outDir, 'event_counts', format=plot_file)

    finally:
        plt.close(fig)


def load_data(root_file: Path) -> dict[str, np.ndarray]:
    """Load distributions from ROOT file produced by optimize_ll.py.

    Args:
        root_file: Path to ROOT file containing distributions

    Returns:
        Dictionary mapping variable names to numpy arrays
    """
    if not root_file.exists():
        print(f"Warning: ROOT file not found: {root_file}")
        return {}

    try:
        with uproot.open(root_file) as file:
            if 'distributions' not in file:
                print(f"Warning: 'distributions' not found in {root_file}")
                return {}

            tree = file['distributions']
            distributions = {}

            # Read all branches
            for key in tree.keys():
                distributions[key] = tree[key].array(library='np')

            return distributions
    except Exception as e:
        print(f"Error loading ROOT file {root_file}: {e}")
        return {}


def plot_comp(
        reco: np.ndarray,
        true: np.ndarray,
        var_name: str,
        xlabel: str,
        outDir: Path,
        label: str,
        bins: int = 100,
        scale: str = 'linear'
         ) -> None:
    """Plot comparison between reconstructed and true distributions.

    Args:
        reconstructed: Array of reconstructed values
        true: Array of true values
        var_name: Name of the variable
        outDir: Output directory for plots
        label: LaTeX label for the final state
        bins: Number of bins for histogram
    """
    # Create figure with subplots
    fig, ax = _make_fig()

    try:
        # Determine common range for both histograms
        xmin = min(reco.min(), true.min())
        xmax = max(reco.max(), true.max())

        # Left plot: Overlay histograms
        ax.hist(reco, bins=bins, range=(xmin, xmax),
                alpha=0.6, label='Reconstructed', color='blue', density=True)
        ax.hist(true, bins=bins, range=(xmin, xmax),
                alpha=0.6, label='True', color='red', density=True)

        set_labels(ax, xlabel, 'Density', right=label)

        ax.set_xlim(xmin, xmax)
        if scale in ['linear', 'log']:
            ax.set_yscale(scale)
            scale = scale.replace('linear', 'lin')
        else:
            raise ValueError(f'{scale = } value is not supported, choose between [lin, log]')
        ax.legend()

        # Save figure
        savefigs(fig, outDir, var_name, suffix='_'+scale, format=plot_file)

    finally:
        plt.close(fig)


def compare_dists(
        old_file: Path,
        optimal_file: Path,
        outDir: Path,
        label: str) -> None:
    """Compare distributions between old (chi2_frac=0.4) and optimal chi2_frac.

    Args:
        old_file: Path to results_old.root
        optimal_file: Path to results_optimal.root
        outDir: Output directory for plots
        label: LaTeX label for the final state
    """
    print("----->[Info] Loading distribution files")
    old_dists = load_data(old_file)
    opt_dists = load_data(optimal_file)

    if not old_dists or not opt_dists:
        print("Warning: Could not load distributions")
        return

    print("----->[Info] Generating distribution comparison plots")

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

    # Create subdirectories for old and optimal
    old_outDir = outDir / 'old'
    opt_outDir = outDir / 'optimal'
    old_outDir.mkdir(parents=True, exist_ok=True)
    opt_outDir.mkdir(parents=True, exist_ok=True)

    # Generate comparison plots for each distribution
    for reco_var, true_var, display_name in comparisons:
        # Check if variables exist
        if reco_var not in old_dists or true_var not in old_dists:
            print(f"Warning: Variables {reco_var} or {true_var} not found in old distributions")
            continue

        if reco_var not in opt_dists or true_var not in opt_dists:
            print(f"Warning: Variables {reco_var} or {true_var} not found in optimal distributions")
            continue

        # Plot old distributions
        for scale in ['linear', 'log']:
            plot_comp(
                old_dists[reco_var],
                old_dists[true_var],
                reco_var,
                display_name,
                old_outDir,
                label,
                scale=scale
            )

            # Plot optimal distributions
            plot_comp(
                opt_dists[reco_var],
                opt_dists[true_var],
                reco_var,
                display_name,
                opt_outDir,
                label,
                scale=scale
            )



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    """Main plotting function"""
    cat, ecm = arg.cat, arg.ecm

    # Get input and output directories
    inDir  = loc.get('OPTIMISATION_RES',   cat, ecm, type=Path)
    outDir = loc.get('PLOTS_OPTIMISATION', cat, ecm, type=Path)

    if not inDir.exists():
        print(f"Error: Input directory not found: {inDir}")
        sys.exit(1)

    # Get processes to plot
    if arg.procs == '':
        procs = [p.name for p in inDir.iterdir() if p.is_dir()]
    else:
        procs = arg.procs.split('-')

    if not procs:
        print(f"Error: No processes found in {inDir}")
        sys.exit(1)

    # Set LaTeX label
    if cat == 'mumu':
        label = r'$Z(\rightarrow\mu^+\mu^-)H'
    elif cat == 'ee':
        label = r'$Z(\rightarrow e^+e^-)H'
    else:
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')

    # Plot results for each process
    for proc in sorted(procs):
        proc_dir = inDir / proc
        results_file = proc_dir / 'results.json'
        out_dir = outDir / proc

        proc_label = proc.replace(f'wzp6_ee_{cat}H', '').replace(f'_ecm{ecm}', '').replace('_H', '')
        if proc_label:
            Label = label + r'(\rightarrow ' + process_label[proc_label] + ')$'
        else:
            Label = label + r')$'

        if not results_file.exists():
            print(f"Warning: Results file not found for {proc}: {results_file}")
            continue

        print(f"\n----->[Info] Processing {proc}")

        # Load results
        results = load_results(results_file)

        if not results:
            print(f"Warning: No results loaded for {proc}")
            continue

        # Create output directory
        out_dir.mkdir(exist_ok=True, parents=True)

        if arg.metrics:
            # Define categories and their output folder names
            categories = {
                'overall':    'overall',
                'one_pair':   'one',
                'multi_pair': 'several'
            }

            # Generate optimization plots for each category
            for category, folder_name in categories.items():
                cat_outDir = out_dir / folder_name
                cat_outDir.mkdir(exist_ok=True, parents=True)

                # Extract arrays for this category
                chi2_fracs, efficiencies, n_correct, n_partial, n_incorrect, n_total = extract_arrays(results, category)

                # Skip if no data for this category
                if len(chi2_fracs) == 0 or np.all(n_total == 0):
                    print(f"Warning: No data for {category} category in {proc}")
                    continue

                # Generate optimization plots
                print(f"----->[Info] Generating {category} plots for {proc}")
                efficiency(chi2_fracs, efficiencies, cat_outDir, Label)
                pairing_composition(
                    chi2_fracs,
                    n_correct, n_partial, n_incorrect, n_total,
                    cat_outDir, Label
                )
                event_counts(
                    chi2_fracs,
                    n_correct, n_partial, n_incorrect,
                    cat_outDir, Label
                )

        if arg.dist:
            # Generate distribution comparison plots
            print(f"----->[Info] Generating distribution comparison plots for {proc}")
            dist_outDir  = out_dir / 'distributions'
            old_file     = proc_dir / 'results_old.root'
            optimal_file = proc_dir / 'results_optimal.root'

            if old_file.exists() and optimal_file.exists():
                compare_dists(old_file, optimal_file, dist_outDir, Label)
            else:
                print(f"Warning: Distribution files not found for {proc}")
                if not old_file.exists():
                    print(f"  Missing: {old_file}")
                if not optimal_file.exists():
                    print(f"  Missing: {optimal_file}")




if __name__ == '__main__':
    try:
        main()
    finally:
        timer(t)
