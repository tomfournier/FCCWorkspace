#!/usr/bin/env python3
"""
Optimizing chi2_recoil_frac for true Z pairing in e+e- -> ZH analysis (Pure Python version)

This script uses uproot and awkward arrays to avoid ROOT's type inference issues.

Usage:
    python3 optimize_ll.py
"""

#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import sys, json, time, uproot

import numpy as np
import awkward as ak

from glob import glob
from tqdm import tqdm
from pathlib import Path

t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    optimize=True,
    description='Optmisation Script'
)
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
from package.config import timer



############################
### OPTIMIZER DEFINITION ###
############################

class Optimizer:
    """Optimize chi2_frac by comparing with MC truth Z pairing"""

    def __init__(
            self,
            proc: str,
            inDir: Path,
            outDir: Path,
            ecm: int = 240,
            nevents: int = -1):
        """Initialize optimizer with pure Python/uproot backend"""

        inFiles = glob(str(inDir / proc / '*'))
        if not inFiles:
            LOGGER.warning(f'No file found in {inDir / proc}\nSearching for {inDir / (proc+".root")} file')
            inFile: Path = inDir / (proc+'.root')
            if  inFile.exists():
                inFiles = [str(inFile)]
            else:
                LOGGER.error(f'No file found in {inDir / (proc+".root")}')
                sys.exit(1)

        self.inFiles = inFiles
        self.outDir = Path(outDir) / proc
        self.outDir.mkdir(parents=True, exist_ok=True)
        self.ecm = ecm

        LOGGER.info(f'Loading data for process {proc}')
        self.load_data(nevents)
        self.results = {}


    def load_data(self, nevents=-1):
        """Load data from ROOT files using uproot (optimized)"""
        # Branch names to load
        branch_names = [
            'n_pair',  # Number of pairs per event
            'mass', 'recoil',  # Reco Z mass and recoil mass
            'Mass', 'Recoil',  # True Z mass and recoil mass
            'leading_mc', 'subleading_mc',  # Reco MC indices
            'Leading_MC', 'Subleading_MC',  # True MC indices
            'leading_p', 'leading_pt', 'leading_theta',           # Reco leading kinematics
            'Leading_p', 'Leading_pt', 'Leading_theta',           # True leading kinematics
            'subleading_p', 'subleading_pt', 'subleading_theta',  # Reco subleading kinematics
            'Subleading_p', 'Subleading_pt', 'Subleading_theta',  # True subleading kinematics
            'zll_p', 'zll_pt', 'zll_theta',  # Reco Z kinematics
            'Zll_p', 'Zll_pt', 'Zll_theta',  # True Z kinematics
        ]

        # Accumulate data per branch
        data = {name: [] for name in branch_names}

        self.total_events = 0
        event_count = 0

        for fpath in self.inFiles:
            with uproot.open(fpath) as file:
                tree = file['events']

                # Get total event count
                self.total_events += file['eventsSelected'].value

                # Read all branches at once
                arrays = tree.arrays(branch_names, library='ak')

                # Determine slice length
                n_entries = len(arrays[branch_names[0]])
                if nevents > 0:
                    n_entries = min(n_entries, nevents - event_count)

                # Slice arrays
                for name in branch_names:
                    arrays[name] = arrays[name][:n_entries]

                # Apply mass cut: remove pairs within 3 GeV of Higgs mass (125 GeV)
                # Only apply to jagged (per-pair) branches
                mass_cut = np.abs(arrays['mass'] - 125) > 3
                jagged_branches = [
                    'mass', 'recoil', 'leading_mc', 'subleading_mc',
                    'leading_p', 'leading_pt', 'leading_theta',
                    'subleading_p', 'subleading_pt', 'subleading_theta',
                    'zll_p', 'zll_pt', 'zll_theta'
                ]

                # Accumulate filtered jagged data and unfiltered flat data
                for name in branch_names:
                    if name in jagged_branches:
                        data[name].append(arrays[name][mass_cut])
                    else:
                        data[name].append(arrays[name])

                event_count += n_entries
                if nevents > 0 and event_count >= nevents:
                    break

        # Concatenate all data and assign them as attributes
        for name in branch_names:
            setattr(self, name, ak.concatenate(data[name]))

        # Recalculate n_pair after filtering (number of pairs per event)
        self.n_pair = ak.num(self.mass)

        # Track actual number of events loaded (not metadata from file)
        self.n_events = len(self.mass)

        # Count events by number of pairs
        n_pair = np.asarray(self.n_pair)
        self.n_zero_pair  = int(np.sum(n_pair == 0))
        self.n_one_pair   = int(np.sum(n_pair == 1))
        self.n_multi_pair = int(np.sum(n_pair >  1))

        LOGGER.info(f'Loaded {self.n_events:,} events (total events in files: {self.total_events:,})')
        LOGGER.info(f'Event breakdown: {self.n_zero_pair:,} with 0 pairs\n'
                    f'                 {self.n_one_pair:,} with 1 pair\n'
                    f'                 {self.n_multi_pair:,} with >1 pair')


    def best_pair_idx(self, dm, drec, frac):
        """
        Find best pair index for each event using precomputed distance matrices.
        Reformulated: chi2_frac * dm + (1-chi2_frac) * drec = drec + chi2_frac * (dm - drec)
        """
        chi2 = drec + frac * (dm - drec)
        return ak.argmin(chi2, axis=1)


    def test_chi2(
            self,
            frac: float,
            dm: np.ndarray,
            drec: np.ndarray):
        """Test chi2_recoil_frac using precomputed distance matrices (fully vectorized)"""

        # Get best pair indices using precomputed distances
        best_idx = self.best_pair_idx(dm, drec, frac)

        # Handle events with 0 pairs: mask them out
        n_pairs = ak.num(self.leading_mc)
        has_pairs = n_pairs > 0

        # Extract best pairs using vectorized awkward masking
        local_idx = ak.local_index(self.leading_mc, axis=1)                # [[0,1,2,...], [0,1,2,...], ...]
        mask_mc1 = local_idx == best_idx[:, None]                                # Create mask for each event
        mask_mc2 = local_idx == best_idx[:, None]
        reco_mc1 = np.asarray(ak.flatten(self.leading_mc[mask_mc1]))     # Extract and flatten
        reco_mc2 = np.asarray(ak.flatten(self.subleading_mc[mask_mc2]))

        # Get true MC indices and pair counts
        true_mc1 = np.asarray(self.Leading_MC)
        true_mc2 = np.asarray(self.Subleading_MC)
        n_pair_array = np.asarray(self.n_pair)

        # Initialize matches array with 0 for all events
        matches = np.zeros(self.n_events, dtype=int)

        # Only compute matches for events that have pairs after filtering
        if len(reco_mc1) > 0:
            match1: np.ndarray = (reco_mc1 == true_mc1[has_pairs]) | (reco_mc1 == true_mc2[has_pairs])
            match2: np.ndarray = (reco_mc2 == true_mc1[has_pairs]) | (reco_mc2 == true_mc2[has_pairs])
            matches[has_pairs] = match1.astype(int) + match2.astype(int)

        # Create masks for different pair categories
        mask_zero  = n_pair_array == 0
        mask_one   = n_pair_array == 1
        mask_multi = n_pair_array >  1

        # Helper function to compute metrics for a category
        def compute_category_stats(matches, mask):
            cat_matches = matches[mask]
            n_pairs = n_pair_array[mask]
            n_correct   = int(np.sum(cat_matches == 2))
            n_partial   = int(np.sum(cat_matches == 1))
            n_incorrect = int(np.sum((cat_matches == 0) & (n_pairs > 0)))  # Only count as incorrect if pairs exist
            n_no_pairs  = int(np.sum(n_pairs == 0))  # Events with no pairs after filtering
            n_total     = int(np.sum(mask))
            efficiency  = n_correct / max(n_total - n_no_pairs, 1)  # Exclude no-pair events from efficiency
            return {
                'n_correct':   n_correct,
                'n_partial':   n_partial,
                'n_incorrect': n_incorrect,
                'n_no_pairs':  n_no_pairs,
                'n_total':     n_total,
                'efficiency':  efficiency,
            }

        # Compute stats for each category
        result = {
            'frac': frac,
            'overall': {
                'efficiency':  int(np.sum(matches == 2)) / max(self.n_events, 1),
                'n_correct':   int(np.sum(matches == 2)),
                'n_partial':   int(np.sum(matches == 1)),
                'n_incorrect': int(np.sum(matches == 0)),
                'n_total':     self.n_events,
            },
            'zero_pair':  compute_category_stats(matches, mask_zero),
            'one_pair':   compute_category_stats(matches, mask_one),
            'multi_pair': compute_category_stats(matches, mask_multi),
        }
        return result

    def extract_distributions(self, chi2_frac: float) -> dict[str, np.ndarray]:
        """Extract kinematic variables for a given chi2_frac value, selecting best pairings"""
        # Precompute distances
        dm = (self.mass - 91.2) ** 2
        drec = (self.recoil - 125) ** 2

        # Get best pair indices
        best_idx = self.best_pair_idx(dm, drec, chi2_frac)

        # Fill None values (from events with 0 pairs) with 0 as placeholder
        best_idx = ak.fill_none(best_idx, 0)

        # Identify events with at least 1 pair after filtering
        n_pairs = ak.num(self.mass)
        has_pairs_mask = np.asarray(n_pairs > 0)

        # Helper function to select values based on best pairing index for each event
        def select_by_idx(array, indices):
            """Select elements from jagged array using indices for each event (vectorized)"""
            if not isinstance(array, ak.Array):
                array = ak.Array(array)

            indices = np.asarray(indices, dtype=int)
            # Create mask where local index matches the best index for each event
            local_idx = ak.local_index(array, axis=1)
            mask = local_idx == indices[:, None]
            # Extract the selected element for each event and flatten
            selected = ak.flatten(array[mask])
            return np.asarray(selected)

        return {
            # Reconstructed leptons (selected based on best pairing)
            'leading_p':     select_by_idx(self.leading_p,     best_idx),
            'leading_pt':    select_by_idx(self.leading_pt,    best_idx),
            'leading_theta': select_by_idx(self.leading_theta, best_idx),

            'subleading_p':     select_by_idx(self.subleading_p,     best_idx),
            'subleading_pt':    select_by_idx(self.subleading_pt,    best_idx),
            'subleading_theta': select_by_idx(self.subleading_theta, best_idx),

            # True leptons (only for events with surviving pairs)
            'Leading_p':     np.asarray(self.Leading_p)[has_pairs_mask],
            'Leading_pt':    np.asarray(self.Leading_pt)[has_pairs_mask],
            'Leading_theta': np.asarray(self.Leading_theta)[has_pairs_mask],

            'Subleading_p':     np.asarray(self.Subleading_p)[has_pairs_mask],
            'Subleading_pt':    np.asarray(self.Subleading_pt)[has_pairs_mask],
            'Subleading_theta': np.asarray(self.Subleading_theta)[has_pairs_mask],

            # Reconstructed Z system (selected based on best pairing)
            'zll_p':     select_by_idx(self.zll_p,     best_idx),
            'zll_pt':    select_by_idx(self.zll_pt,    best_idx),
            'zll_theta': select_by_idx(self.zll_theta, best_idx),

            # True Z system (only for events with surviving pairs)
            'Zll_p':     np.asarray(self.Zll_p)[has_pairs_mask],
            'Zll_pt':    np.asarray(self.Zll_pt)[has_pairs_mask],
            'Zll_theta': np.asarray(self.Zll_theta)[has_pairs_mask],

            # Pair variables (selected based on best pairing)
            'mass':   select_by_idx(self.mass,   best_idx),
            'recoil': select_by_idx(self.recoil, best_idx),
            'Mass':   np.asarray(self.Mass)[has_pairs_mask],
            'Recoil': np.asarray(self.Recoil)[has_pairs_mask]
        }

    def save_distributions(self, chi2_frac: float, filename: str):
        """Save kinematic distributions to a ROOT file for a given chi2_frac"""
        # Extract distributions
        distributions = self.extract_distributions(chi2_frac)

        # Create output file path
        outFile = self.outDir / filename
        outFile.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for ROOT file (convert all to numpy arrays)
        output_data: dict[str, np.ndarray] = {}
        for key, values in distributions.items():
            if isinstance(values, ak.Array):
                values = np.asarray(values)
            # Ensure proper dtype (float32 or float64)
            if not np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32)
            # Flatten to 1D if needed
            if values.ndim > 1:
                values = values.flatten()
            output_data[key] = values

        # Write to ROOT file as structured array / TTree
        with uproot.recreate(str(outFile)) as file:
            # Create a structured numpy array from the dictionary
            dtype = [(key, values.dtype) for key, values in output_data.items()]
            # Find the common length (should be the same for all arrays)
            n_events = len(next(iter(output_data.values())))

            # Create structured array
            structured_data = np.zeros(n_events, dtype=dtype)
            for key, values in output_data.items():
                structured_data[key] = values

            # Write as ROOT tree
            file["distributions"] = structured_data

        LOGGER.info(f'Variables distribution saved to {outFile}')


    def optimize(
            self,
            chi2_values: list[float] | np.ndarray,
            mass: int | float = 91.2,
            recoil: int | float = 125):
        """Run optimization loop over chi2_recoil_frac values (with precomputed distances)"""

        # Precompute squared distances once to avoid recalculation in each iteration
        dm = (self.mass - mass) ** 2
        drec = (self.recoil - recoil) ** 2

        for chi2_frac in tqdm(chi2_values):
            result = self.test_chi2(chi2_frac, dm, drec)
            self.results[chi2_frac] = result

        return self.results


    def save_results(self):
        """Save results to JSON with breakdown by number of pairs"""
        outFile = self.outDir / 'results.json'

        json_results = {}
        for chi2_frac, result in self.results.items():
            json_results[f"{chi2_frac:.2f}"] = {
                'chi2_frac': result['frac'],
                'overall': {
                    'efficiency':  result['overall']['efficiency'],
                    'n_correct':   result['overall']['n_correct'],
                    'n_partial':   result['overall']['n_partial'],
                    'n_incorrect': result['overall']['n_incorrect'],
                    'n_total':     result['overall']['n_total']
                },
                'zero_pair': {
                    'efficiency':  result['zero_pair']['efficiency'],
                    'n_correct':   result['zero_pair']['n_correct'],
                    'n_partial':   result['zero_pair']['n_partial'],
                    'n_incorrect': result['zero_pair']['n_incorrect'],
                    'n_no_pairs':  result['zero_pair']['n_no_pairs'],
                    'n_total':     result['zero_pair']['n_total']
                },
                'one_pair': {
                    'efficiency':  result['one_pair']['efficiency'],
                    'n_correct':   result['one_pair']['n_correct'],
                    'n_partial':   result['one_pair']['n_partial'],
                    'n_incorrect': result['one_pair']['n_incorrect'],
                    'n_no_pairs':  result['one_pair']['n_no_pairs'],
                    'n_total':     result['one_pair']['n_total']
                },
                'multi_pair': {
                    'efficiency':  result['multi_pair']['efficiency'],
                    'n_correct':   result['multi_pair']['n_correct'],
                    'n_partial':   result['multi_pair']['n_partial'],
                    'n_incorrect': result['multi_pair']['n_incorrect'],
                    'n_no_pairs':  result['multi_pair']['n_no_pairs'],
                    'n_total':     result['multi_pair']['n_total']
                }
            }

        outFile.write_text(json.dumps(json_results, indent=4))
        LOGGER.info(f'Results saved at {outFile}')
        return outFile



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    """Main analysis function"""
    cat, ecm, nevents = arg.cat, arg.ecm, arg.nevents

    inDir = loc.get('OPTIMISATION_TEST', cat, ecm, type=Path)
    outDir = loc.get('OPTIMISATION_RES', cat, ecm, type=Path)

    if not inDir.exists():
        LOGGER.error(f'Input directory not found at {inDir}')
        sys.exit(1)

    if arg.procs == '':
        procs = [Path(p).name for p in glob(str(inDir / '*'))]
    else:
        procs = arg.procs.split('-')

    for proc in procs:
        optimizer = Optimizer(proc, inDir, outDir, ecm, nevents=nevents)
        optimizer.optimize(np.arange(0, 1+arg.incr, arg.incr))
        optimizer.save_results()

        # Find optimal chi2_frac
        best_frac = max(optimizer.results.items(), key=lambda x: x[1]['overall']['efficiency'])[0]
        LOGGER.info(f'Optimal chi2_frac: {best_frac:.2f}')

        # Save distributions for chi2_frac = 0.6
        LOGGER.info('Saving variables distribution for chi2_frac = 0.6')
        optimizer.save_distributions(0.6, 'results_old.root')

        # Save distributions for optimal chi2_frac
        LOGGER.info(f'Saving variables distribution for optimal chi2_frac = {best_frac:.2f}')
        optimizer.save_distributions(best_frac, 'results_optimal.root')


if __name__ == '__main__':
    try:
        main()
    finally:
        timer(t)
