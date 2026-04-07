#!/usr/bin/env python3
"""
Optimizing chi2_recoil_frac for true Z pairing in e+e- -> ZH analysis (Pure Python version)

This script uses uproot and awkward arrays to avoid ROOT's type inference issues.

Usage:
    python3 optimize_ll.py
"""

###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

import sys, json, time, uproot

import numpy as np
import awkward as ak

from glob import glob
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from package.userConfig import loc
from package.config import timer

t = time.time()



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser()
parser.add_argument('--cat', choices=['ee', 'mumu'], default='mumu')
parser.add_argument('--ecm', choices=[240, 365], type=int, default=240)
parser.add_argument('--procs', type=str, default='')
parser.add_argument('--nevents', type=int, default=-1)
parser.add_argument('--incr', type=float, default=0.1)
arg = parser.parse_args()



############################
### OPTIMIZER DEFINITION ###
############################

class Optimizer:
    """Optimize chi2_frac by comparing with MC truth Z pairing"""

    def __init__(
            self,
            proc: str,
            inDir: str,
            outDir: str,
            ecm: int = 240,
            nevents: int = -1):
        """Initialize optimizer with pure Python/uproot backend"""

        inFiles = glob(str(inDir / proc / '*'))
        if not inFiles:
            print(f"Warning: No files found in {inDir / proc}/")
            sys.exit(1)

        self.inFiles = inFiles
        self.outDir = Path(outDir) / proc
        self.outDir.mkdir(parents=True, exist_ok=True)
        self.ecm = ecm

        print('----->[Info] Loading data from ROOT files')
        self.load_data(nevents)
        self.results = {}


    def load_data(self, nevents=-1):
        """Load data from ROOT files using uproot (optimized)"""
        # Branch names to load - all kinematic variables are pre-computed
        branch_names = [
            'mass', 'recoil', 'Mass', 'Recoil',
            'mc1', 'mc2', 'MC1', 'MC2',
            # Reconstructed leptons l1, l2
            'l1_e', 'l1_p', 'l1_pt', 'l1_theta',
            'l2_e', 'l2_p', 'l2_pt', 'l2_theta',
            # True leptons lep1, lep2
            'lep1_e', 'lep1_p', 'lep1_pt', 'lep1_theta',
            'lep2_e', 'lep2_p', 'lep2_pt', 'lep2_theta',
            # Reconstructed Z system
            'z_e', 'z_p', 'z_pt', 'z_theta',
            # True Z system
            'Z_e', 'Z_p', 'Z_pt', 'Z_theta',
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

                # Read all branches at once (faster than individual reads)
                arrays = tree.arrays(branch_names, library='ak')

                # Determine slice length
                n_entries = len(arrays[branch_names[0]])
                if nevents > 0:
                    n_entries = min(n_entries, nevents - event_count)

                # Accumulate sliced data
                for name in branch_names:
                    data[name].append(arrays[name][:n_entries])

                event_count += n_entries
                if nevents > 0 and event_count >= nevents:
                    break

        # Concatenate all data and assign as attributes
        for name in branch_names:
            setattr(self, name, ak.concatenate(data[name]))

        # Track actual number of events loaded (not metadata from file)
        self.n_events = len(self.mass)

        print(f"----->[Info] Loaded {self.n_events:,} events (total events in files: {self.total_events:,})")


    def best_pair_idx(self, dm, drec, chi2_frac):
        """
        Find best pair index for each event using precomputed distance matrices.
        Reformulated: chi2_frac * dm + (1-chi2_frac) * drec = drec + chi2_frac * (dm - drec)
        """
        chi2 = drec + chi2_frac * (dm - drec)
        return ak.argmin(chi2, axis=1)


    def test_chi2(
            self,
            frac: float,
            dm: np.ndarray,
            drec: np.ndarray):
        """Test chi2_recoil_frac using precomputed distance matrices (fully vectorized)"""

        # Get best pair indices using precomputed distances
        best_idx = self.best_pair_idx(dm, drec, frac)

        # Extract best pairs using vectorized awkward masking
        local_idx = ak.local_index(self.mc1, axis=1)  # [[0,1,2,...], [0,1,2,...], ...]
        mask_mc1 = local_idx == best_idx[:, None]           # Create mask for each event
        mask_mc2 = local_idx == best_idx[:, None]
        reco_mc1 = ak.flatten(self.mc1[mask_mc1])     # Extract and flatten
        reco_mc2 = ak.flatten(self.mc2[mask_mc2])
        reco_mc1 = np.asarray(reco_mc1)
        reco_mc2 = np.asarray(reco_mc2)

        # Get true MC indices
        true_mc1 = np.asarray(self.MC1)
        true_mc2 = np.asarray(self.MC2)

        # Vectorized matching: count how many reconstructed leptons match truth
        match1 = (reco_mc1 == true_mc1) | (reco_mc1 == true_mc2)
        match2 = (reco_mc2 == true_mc1) | (reco_mc2 == true_mc2)

        # Sum matches per event (0, 1, or 2)
        matches = match1.astype(int) + match2.astype(int)

        # Count results using vectorized operations
        n_correct   = int(np.sum(matches == 2))
        n_partial   = int(np.sum(matches == 1))
        n_incorrect = int(np.sum(matches == 0))

        result = {
            'frac': frac,
            'efficiency':  n_correct / max(self.n_events, 1),
            'n_correct':   n_correct,
            'n_partial':   n_partial,
            'n_incorrect': n_incorrect,
            'n_total': self.n_events,
        }
        return result

    def extract_distributions(self, chi2_frac: float):
        """Extract kinematic variables for a given chi2_frac value, selecting best pairings"""
        # Precompute distances
        dm = (self.mass - 91.2) ** 2
        drec = (self.recoil - 125) ** 2

        # Get best pair indices
        best_idx = self.best_pair_idx(dm, drec, chi2_frac)

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
            'l1_e':     select_by_idx(self.l1_e,     best_idx),
            'l1_p':     select_by_idx(self.l1_p,     best_idx),
            'l1_pt':    select_by_idx(self.l1_pt,    best_idx),
            'l1_theta': select_by_idx(self.l1_theta, best_idx),

            'l2_e':     select_by_idx(self.l2_e,     best_idx),
            'l2_p':     select_by_idx(self.l2_p,     best_idx),
            'l2_pt':    select_by_idx(self.l2_pt,    best_idx),
            'l2_theta': select_by_idx(self.l2_theta, best_idx),

            # True leptons (not affected by pairing selection)
            'lep1_e':     np.asarray(self.lep1_e),
            'lep1_p':     np.asarray(self.lep1_p),
            'lep1_pt':    np.asarray(self.lep1_pt),
            'lep1_theta': np.asarray(self.lep1_theta),

            'lep2_e':     np.asarray(self.lep2_e),
            'lep2_p':     np.asarray(self.lep2_p),
            'lep2_pt':    np.asarray(self.lep2_pt),
            'lep2_theta': np.asarray(self.lep2_theta),

            # Reconstructed Z system (selected based on best pairing)
            'z_e':     select_by_idx(self.z_e, best_idx),
            'z_p':     select_by_idx(self.z_p, best_idx),
            'z_pt':    select_by_idx(self.z_pt, best_idx),
            'z_theta': select_by_idx(self.z_theta, best_idx),

            # True Z system (not affected by pairing selection)
            'Z_e':     np.asarray(self.Z_e),
            'Z_p':     np.asarray(self.Z_p),
            'Z_pt':    np.asarray(self.Z_pt),
            'Z_theta': np.asarray(self.Z_theta),

            # Pair variables (selected based on best pairing)
            'mass':   select_by_idx(self.mass,   best_idx),
            'recoil': select_by_idx(self.recoil, best_idx),
            'Mass':   np.asarray(self.Mass),
            'Recoil': np.asarray(self.Recoil)
        }

    def save_distributions(self, chi2_frac: float, filename: str):
        """Save kinematic distributions to a ROOT file for a given chi2_frac"""
        # Extract distributions
        distributions = self.extract_distributions(chi2_frac)

        # Create output file path
        outFile = self.outDir / filename
        outFile.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for ROOT file (convert all to numpy arrays)
        output_data = {}
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

        print(f"----->[Info] Distributions saved to: {outFile}")


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
        """Save results to JSON"""
        outFile = self.outDir / 'results.json'

        json_results = {}
        for chi2_frac, result in self.results.items():
            json_results[f"{chi2_frac:.2f}"] = {
                'chi2_frac':   result['frac'],
                'efficiency':  result['efficiency'],
                'n_correct':   result['n_correct'],
                'n_partial':   result['n_partial'],
                'n_incorrect': result['n_incorrect'],
                'n_total':     result['n_total']
            }

        outFile.write_text(json.dumps(json_results, indent=4))
        print(f"----->[Info] Results saved: {outFile}")
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
        print(f"Error: Input directory not found: {inDir}")
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
        best_frac = max(optimizer.results.items(), key=lambda x: x[1]['efficiency'])[0]
        print(f"----->[Info] Optimal chi2_frac: {best_frac:.2f}")

        # Save distributions for chi2_frac = 0.4
        print("----->[Info] Saving distributions for chi2_frac = 0.4")
        optimizer.save_distributions(0.4, 'results_old.root')

        # Save distributions for optimal chi2_frac
        print(f"----->[Info] Saving distributions for optimal chi2_frac = {best_frac:.2f}")
        optimizer.save_distributions(best_frac, 'results_optimal.root')


if __name__ == '__main__':
    try:
        main()
    finally:
        timer(t)
