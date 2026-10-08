'''Core configuration for the FCC-ee ZH cross-section analysis.

Provides:
- Feature set for BDT training: `input_vars`.
- Decay mode enumerations: `Z_DECAYS`, `H_DECAYS`, `H_DECAYS_WITH_INV`, `H_DECAYS_ALL`, `QUARKS`
    plus lowercase aliases for backward compatibility.
- Color palettes for ROOT and matplotlib: `colors`, `h_colors`, `modes_color`.
- Physics and axis labels (ROOT TLatex and LaTeX): `labels`, `h_labels`,
    `vars_label`, `vars_xlabel`, `modes_label`, `process_label`.
- Process builders:
    - `get_process_dict()`: Simple process dictionary builder with optional filtering.
    - `get_process_list()`: Full-featured process builder with signal/background handling.
- Background/signal process construction for analysis workflows.
- Utilities: `warning()` for formatted exceptions and `timer()` for timing output.

Conventions:
- Process naming follows FCC patterns, e.g. ``wzp6_ee_{z}H_H{h}_ecm{ecm}``,
    ``p8_ee_WW_ecm{ecm}``.
- Labels use ROOT TLatex syntax for ROOT displays and LaTeX for matplotlib.
- Units are appended in `vars_xlabel` (e.g., GeV, GeV^2).

Usage:
- Simple process dictionary: ``get_process_dict(procs=['ZH','WW'], ecm=365)``.
- Full analysis workflow: ``get_process_list(cat='mumu', ecm=240, train=True)``.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from typing import Sequence, Union

from constants import (
    Z_DECAYS, H_DECAYS, QUARKS,
    H_DECAYS_ALL, H_DECAYS_WITH_INV
)
from logger import get_logger
LOGGER = get_logger(__name__)



##########################
### PROCESSES FUNCTION ###
##########################

def parse_sample_selection(selection: str) -> dict[str, dict[str, float | int]]:
    '''Parse CLI sample selections into process overrides.

    Entries are separated by ``:``. Each entry is either a sample name, or a
    sample name followed by its fraction and chunk count:
    ``sample`` or ``sample,fraction,chunks``.
    '''
    if not selection:
        return {}

    samples = {}
    for entry in selection.split(':'):
        entry = entry.strip()
        if not entry:
            raise ValueError('Sample selections cannot contain empty entries.')

        fields = entry.split(',')
        if len(fields) not in (1, 3) or not fields[0]:
            raise ValueError(f'Invalid sample selection {entry!r}; expected '
                             'sample or sample,fraction,chunks.')

        sample = fields[0]
        if len(fields) == 1:
            samples[sample] = {}
            continue

        try:
            fraction = float(fields[1])
            chunks = int(fields[2])
        except ValueError as error:
            raise ValueError(f'Invalid values in sample selection {entry!r}; '
                             'fraction must be a number and chunks an integer.') from error
        if chunks < 1: raise ValueError(f'Chunk count must be at least 1 in {entry!r}.')
        if fraction < 0 or fraction > 1: raise ValueError('fraction must be between 0 and 1')
        samples[sample] = {'fraction': fraction, 'chunks': chunks}

    return samples


def parse_sample_exclusion(exclusion: str) -> set[str]:
    '''Parse colon-separated sample names for the exclusion option.'''
    return {sample.strip() for sample in exclusion.split(':') if sample.strip()}


def get_process_dict(
    procs:    Union[Sequence[str], None] = None,
    ecm: int = 240,
    z_decays: Union[Sequence[str], None] = None,
    h_decays: Union[Sequence[str], None] = None,
    H_decays: Union[Sequence[str], None] = None,
    quarks:   Union[Sequence[str], None] = None,
) -> dict[str, tuple[str, ...]]:
    '''Generate process dictionary with optional filtering and custom decay modes.

    Simple process builder for creating FCC sample dictionaries.
    Can use defaults (cached) or custom decay modes. Optionally filters to specific process keys.
    Returns process key -> sample names mapping (e.g., 'ZH' -> ('wzp6_ee_bbH_Hbb_ecm240', ...)).

    Args:
        procs: Process keys to include. If None, returns all available processes.
        z_decays: Z decay modes. Uses Z_DECAYS if None.
        h_decays: Higgs decay modes (no invisible). Uses H_DECAYS if None.
        H_decays: Higgs decay modes (with invisible). Uses H_DECAYS_WITH_INV if None.
        quarks: Quark channels. Uses QUARKS if None.
        ecm: Center-of-mass energy in GeV (default 240).

    Returns:
        Dictionary mapping process keys to tuples of FCC sample names.

    Examples:
        >>> get_process_dict()  # All processes, default decays, 240 GeV
        >>> get_process_dict(procs=['ZH', 'WW'], ecm=365)  # Filtered, 365 GeV
        >>> get_process_dict(h_decays=['bb', 'cc'])  # Custom Higgs decays
    '''
    z_set = Z_DECAYS if z_decays is None else tuple(z_decays)
    h_set = H_DECAYS if h_decays is None else tuple(h_decays)
    H_set = H_DECAYS_WITH_INV if H_decays is None else tuple(H_decays)
    q_set = QUARKS   if quarks   is None else tuple(quarks)

    processes = {
        # All signals for the Z and Higgs exclusive decay
        'ZH':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_set for y in h_set),

        # All signals for a specific Z decays and Higgs exclusive decay
        'ZeeH':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}' for y in h_set),
        'ZmumuH': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_set),
        'ZqqH':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in q_set for y in h_set),

        # All signals for the Z and Higgs exclusive decay (Include invisible decay)
        'zh':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_set for y in H_set),

        # All signals for a specific Z decays and Higgs exclusive decay (Include invisible decay)
        'zeeh':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}' for y in H_set),
        'zmumuh': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in H_set),
        'zqqh':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in q_set for y in H_set),

        # Diboson production e+e- -> VV (V = W or Z)
        'WW':     (f'p8_ee_WW_ecm{ecm}', f'p8_ee_WW_ee_ecm{ecm}', f'p8_ee_WW_mumu_ecm{ecm}'),
        'ZZ':     (f'p8_ee_ZZ_ecm{ecm}',),

        # 2 fermion production e+e- -> ff
        'Zgamma': (f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
                   f'wzp6_ee_tautau_ecm{ecm}',        f'wzp6_ee_qq_ecm{ecm}'),

        # Rare processes: photon induced, diphoton and nunuZ processes
        'Rare':   (f'wzp6_gammae_eZ_Zee_ecm{ecm}',    f'wzp6_egamma_eZ_Zee_ecm{ecm}',
                   f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',  f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
                   f'wzp6_gammae_eZ_Zqq_ecm{ecm}',    f'wzp6_egamma_eZ_Zqq_ecm{ecm}',
                   f'wzp6_gaga_ee_60_ecm{ecm}',       f'wzp6_gaga_mumu_60_ecm{ecm}',
                   f'wzp6_gaga_tautau_60_ecm{ecm}',   f'wzp6_ee_nuenueZ_ecm{ecm}'),
    }
    if ecm == 365:
        # Include e+e- -> tt process for ecm = 365 GeV
        processes['tt'] = ('wzp6_ee_WbWb_ecm365',)

    if procs:
        return {proc: processes[proc] for proc in procs if proc in processes}
    return processes


def get_bdt_modes(cat: str, ecm: int) -> dict[str, list[str]]:
    '''Generate the signal and background samples used by the BDT.

    Args:
        cat: Analysis category ('ee', 'mumu', or 'qq').
        ecm: Center-of-mass energy in GeV.

    Returns:
        Dictionary mapping BDT mode names to FCC sample names.
    '''
    if cat not in {'ee', 'mumu', 'qq'}:
        raise ValueError(f'{cat = } is not a valid category. Use [ee, mumu, qq].')

    signal_samples = ([
        f'wzp6_ee_{cat}H_ecm{ecm}'] if cat in {'ee', 'mumu'} else [f'wzp6_ee_{quark}H_H{decay}_ecm{ecm}'
                                                                   for quark in QUARKS for decay in H_DECAYS])

    modes = {}
    modes[f'Z{cat}H'] = signal_samples
    modes[f'WW{cat}'] = [f'p8_ee_WW_ecm{ecm}' if cat == 'qq' else f'p8_ee_WW_{cat}_ecm{ecm}']
    modes['ZZ']       = [f'p8_ee_ZZ_ecm{ecm}']
    modes[f'Z{cat}']  = [f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat == 'ee' else f'wzp6_ee_{cat}_ecm{ecm}']
    modes[f'egamma_{cat}'] = [f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}']
    modes[f'gammae_{cat}'] = [f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}']

    if cat != 'qq': modes[f'gaga_{cat}'] = [f'wzp6_gaga_{cat}_60_ecm{ecm}']
    if cat == 'qq' and ecm == 365: modes['ttbar'] = ['wzp6_ee_WbWb_ecm365']

    return modes


def get_process_list(
    cat: str,
    ecm: int,
    z_decays: tuple[str, ...] = Z_DECAYS,
    h_decays: tuple[str, ...] = H_DECAYS_ALL,
    quarks: tuple[str, ...] = QUARKS,
    train: bool = False,
    batch: bool = False,
    onlysig: bool = False,
    onlybkg: bool = False,
    frac: dict[str, float] | None = None,
    chunks: dict[str, int] | None = None,
    include: dict[str, dict] | None = None,
    exclude: set[str] | None = None,
    all_train_sig: bool = True
) -> dict[str, dict[str, float | int]]:
    '''Generate analysis-ready process dictionary with signals and backgrounds.

    Full-featured process builder for analysis workflows. Combines signal and background
    samples with event counts and fractions. Training mode uses simplified samples.
    Supports filtering, custom overrides, and batch mode scaling.

    Args:
        cat: Category ('ee', 'mumu', 'qq').
        ecm: Center-of-mass energy in GeV (240 or 365).
        z_decays: Z decay modes (non-training mode only; training uses defaults).
        h_decays: Higgs decay modes (non-training mode only; training uses defaults).
        train: If True, use training-mode samples (category-specific backgrounds).
        batch: If True, scale chunk sizes for batch processing.
        onlysig: Return only signal processes (mutually exclusive with onlybkg).
        onlybkg: Return only background processes (mutually exclusive with onlysig).
        frac: Custom fractions by sample name (overrides defaults).
        chunks: Custom event chunk counts by sample name (overrides defaults).
        include: Additional processes to add, dict with 'sig' and/or 'bkg' keys.
        exclude: Set of sample names to exclude from output.

    Returns:
        Dictionary mapping sample names to {'fraction': float, 'chunks': int}.

    Raises:
        ValueError: If onlysig and onlybkg are both True.
    '''
    # Initialize optional parameters
    frac    = frac    or {}
    chunks  = chunks  or {}
    include = include or {}
    exclude = exclude or set()

    # Validate conflicting options
    if onlysig and onlybkg:
        raise ValueError('Cannot set both onlysig and onlybkg to True. Choose one.')

    if train:
        if cat in ['ee', 'mumu']:
            sigs = [f'wzp6_ee_{cat}H_ecm{ecm}']
            if all_train_sig:
                sigs += [f'wzp6_ee_{cat}H_H{y}_ecm{ecm}' for y in h_decays if 'noInv' not in y]
        elif cat == 'qq':
            sigs = [f'wzp6_ee_{x}H_ecm{ecm}' for x in quarks]
            if all_train_sig:
                sigs += [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in quarks for y in h_decays if 'noInv' not in y]
        else:
            raise ValueError(
                f'{cat} is not a valid category. Use [ee, mumu, qq].')
    else:
        sigs = [
            f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays]

    small, middle, big = ((5, 5, 10) if batch else (1, 5, 10)) if train \
        else ((5, 20, 30) if batch else (1, 5, 10))
    common = {f'p8_ee_ZZ_ecm{ecm}': {'frac': 0.25 if cat == 'qq' else 1, 'nb': middle}}
    if not train or cat == 'qq':
        common[f'p8_ee_WW_ecm{ecm}'] = {'frac': (
            0.3 if ecm == 240 else 1) if train else (0.1 if cat == 'qq' else 1), 'nb': big}

    category_specific: dict[str, dict[str, float | int]] = {
        'ee': {
            f'p8_ee_WW_ee_ecm{ecm}':           {'frac': 1, 'nb': middle},
            f'wzp6_ee_ee_Mee_30_150_ecm{ecm}': {'frac': 1, 'nb': big},
            f'wzp6_egamma_eZ_Zee_ecm{ecm}':    {'frac': 1, 'nb': middle},
            f'wzp6_gammae_eZ_Zee_ecm{ecm}':    {'frac': 1, 'nb': middle},
            f'wzp6_gaga_ee_60_ecm{ecm}':       {'frac': 1, 'nb': middle}},
        'mumu': {
            f'p8_ee_WW_mumu_ecm{ecm}':        {'frac': 1, 'nb': middle},
            f'wzp6_ee_mumu_ecm{ecm}':         {'frac': 1, 'nb': big},
            f'wzp6_egamma_eZ_Zmumu_ecm{ecm}': {'frac': 1, 'nb': middle},
            f'wzp6_gammae_eZ_Zmumu_ecm{ecm}': {'frac': 1, 'nb': middle},
            f'wzp6_gaga_mumu_60_ecm{ecm}':    {'frac': 1, 'nb': middle}},
        'qq': {
            f'wzp6_ee_qq_ecm{ecm}':         {'frac': 1,   'nb': middle},
            f'wzp6_egamma_eZ_Zqq_ecm{ecm}': {'frac': 1,   'nb': middle},
            f'wzp6_gammae_eZ_Zqq_ecm{ecm}': {'frac': 1,   'nb': middle}},
    }
    if ecm == 365:
        category_specific['qq'].update({
            'wzp6_ee_WbWb_ecm365': {'frac': 1, 'nb': small},
            'p8_ee_tt_ecm365':     {'frac': 1, 'nb': small}})

    if train:
        bkgs = {**common, **category_specific.get(cat, {})}
    else:
        bkgs = {**common, **category_specific.get(cat, {}),
                f'wzp6_ee_tautau_ecm{ecm}':      {'frac': 1, 'nb': small},
                f'wzp6_gaga_tautau_60_ecm{ecm}': {'frac': 1, 'nb': small},
                f'wzp6_ee_nuenueZ_ecm{ecm}':     {'frac': 1, 'nb': small}}
        if cat == 'qq':
            bkgs.update({
                f'p8_ee_WW_ee_ecm{ecm}':           {'frac': 1,   'nb': middle},
                f'p8_ee_WW_mumu_ecm{ecm}':         {'frac': 1,   'nb': middle},
                f'wzp6_ee_ee_Mee_30_150_ecm{ecm}': {'frac': 0.1, 'nb': middle},
                f'wzp6_ee_mumu_ecm{ecm}':          {'frac': 0.1, 'nb': middle}})

    # Build signal dict with custom overrides
    process_sig = {s: {'fraction': frac.get(s, 1), 'chunks': chunks.get(s, 1)}
                   for s in sigs if (s not in exclude and 'all' not in exclude)}

    # Build background dict with custom overrides
    process_bkg = {b: {'fraction': frac.get(b, v['frac']), 'chunks': chunks.get(b, v['nb'])}
                   for b, v in bkgs.items() if (b not in exclude and 'all' not in exclude)}

    # Apply custom inclusions. The nested form is retained for programmatic
    # callers; the flat form is convenient for CLI sample selection.
    if 'sig' in include or 'bkg' in include:
        process_sig = {**process_sig, **include.get('sig', {})}
        process_bkg = {**process_bkg, **include.get('bkg', {})}
    else:
        for sample, override in include.items():
            if sample in sigs:
                defaults = {'fraction': 1, 'chunks': 1}
                process_sig[sample] = {**defaults, **override}
            elif sample in bkgs:
                defaults = {'fraction': bkgs[sample]['frac'], 'chunks': bkgs[sample]['nb']}
                process_bkg[sample] = {**defaults, **override}
            else:
                raise ValueError(f'Cannot classify included sample {sample!r} as signal or background.')

    # Return requested subset
    if onlysig: return process_sig
    if onlybkg: return process_bkg
    return {**process_sig, **process_bkg}
