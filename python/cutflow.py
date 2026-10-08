'''Cutflow and selection efficiency analysis with visualization.

Provides:
- Cut expression parsing and variable extraction: `branches_from_cuts()`.
- Event count aggregation across cut steps: `CutFlow()`.
- Decay-mode efficiency analysis: `CutFlowDecays()`, `Efficiency()`.
- ASCII table generation: `write_table()`.
- Integration with ROOT graphics backend for publication plots.

Functions:
- `branches_from_cuts()`: Extract variable names referenced in cut filter expressions using regex.
- `write_table()`: Format and write aligned ASCII tables with configurable column widths.
- `CutFlow()`: Render stacked cutflow histogram with signal overlay, significance values, and yields table.
- `CutFlowDecays()`: Plot efficiency curves (normalized to first cut) for each Higgs decay mode.
- `Efficiency()`: Generate efficiency summary plots with pull diagrams and per-decay tables.

Conventions:
- Cut steps indexed sequentially (cut0, cut1, ...) matching provided labels.
- Yields computed with luminosity and cross-section scaling when available.
- Uncertainties treated as Poisson (√N) on raw counts, scaled linearly.
- Significance computed per cut as S/√(S+B) where S=signal, B=total background.
- Efficiency normalized to first cut (cut0) for decay mode comparisons (%).
- Channel-specific plots (ee/mumu) optionally combined into totals (tot).
- ROOT histograms stored in flow dict with keys: flow[process]['hist'] = [TH1, ...].

Usage:
- Analyze event selection efficiency by computing cumulative event counts per cut step.
- Generate publication-ready cutflow plots with stacked backgrounds and signal overlay.
- Validate selection consistency across Higgs decay modes via efficiency tables and plots.
- Export yield summaries and efficiency pulls as both plots and tabular data.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import copy

from tqdm import tqdm

from tools.utils import get_df
from tools.process import getMetaInfo
from logger import get_logger

LOGGER = get_logger(__name__)


######################
### MAIN FUNCTIONS ###
######################

# ___________________________________
def branches_from_cuts(
    cuts: dict[str, dict[str, str]],
    variables: list
) -> list:
    '''Extract variables used in cut expressions.

    Scans all cut expressions and identifies which variables from the provided
    list are actually referenced. Results are sorted for deterministic output.

    Args:
        cuts (dict[str, dict[str, str]]): Dictionary mapping selection names to cut definitions (cut_name -> expression).
        variables (list): List of variable names to search for.

    Returns:
        list: Sorted list of variables found in cut expressions.
    '''
    import re
    used = set()
    for sel_cuts in cuts.values():
        for expr in sel_cuts.values():
            if not expr:  # Skip empty expressions
                continue
            for var in variables:
                # Match whole words only using regex word boundaries
                if re.search(r'\b' + re.escape(var) + r'\b', expr):
                    used.add(var)
    return sorted(used)


def get_cutflow(
    inDir: str,
    outDir: str,
    cat: str,
    sels: list[str],
    procs: list[str],
    procs_decays: list[str],
    processes: dict[str, list[str]],
    colors: dict[str, dict[str, str]],
    legend: dict[str, dict[str, str]],
    cuts: dict[str, dict[str, str]],
    cuts_label: dict[str, dict[str, str]],
    z_decays: list[str],
    H_decays: list[str],
    format: list[str] = ['png'],
    ecm: int = 240,
    lumi: float = 10.8,
    sig_scale: float = 1.0,
    branches: list[str] = [],
    scaled: bool = True,
    tot: bool = False,
    json_file: bool = False,
    loc_json: str = ''
) -> None:
    '''Compute cutflows from parquet/ROOT inputs and render plots.

    Args:
        inDir (str): Directory containing processed event files.
        outDir (str): Base output directory for plots and tables.
        cat (str): Channel tag (``ee`` or ``mumu``).
        sels (list[str]): Selection names to evaluate.
        procs (list[str]): Processes for plotting; signal first.
        procs_decays (list[str]): Processes including decay-specific entries.
        processes (dict[str, list[str]]): Map of process -> list of sample identifiers.
        colors (dict[str, dict[str, str]]): Color mapping per process for styling.
        legend (dict[str, dict[str, str]]): Legend labels per process.
        cuts (dict[str, dict[str, str]]): Cut definitions per selection.
        cuts_label (dict[str, dict[str, str]]): Human-readable labels per cut step.
        z_decays (list[str]): Z decay modes included.
        H_decays (list[str]): Higgs decay modes included.
        format (list[str], optional): Output image formats. Defaults to ['png'].
        ecm (int, optional): Center-of-mass energy. Defaults to 240.
        lumi (float, optional): Integrated luminosity in ab^-1. Defaults to 10.8.
        sig_scale (float, optional): Scale factor applied to signal yields. Defaults to 1.0.
        branches (list[str], optional): Optional list of columns to read from files. Defaults to [].
        scaled (bool, optional): If True, scale yields by cross section and luminosity. Defaults to True.
        tot (bool, optional): If True, also produce totals across lepton categories. Defaults to False.
        json_file (bool, optional): If True, persist intermediate JSON summaries. Defaults to False.
        loc_json (str, optional): Path where JSON snapshots are stored. Defaults to ''.
    '''

    import numpy as np
    from plots.python.helper import (
        find_sample_files,
        is_there_events,
        get_processed,
        get_count,
        get_flows,
        dump_json
    )
    from plotting.plotting import CutFlow, CutFlowDecays

    # Initialize event and file list dictionaries
    events, file_list = {}, {}
    # Update process names to channel-specific variants if not computing totals
    if not tot:
        procs[0] = f'Z{cat}H'
        procs_decays[0] = f'z{cat}h'

    # Collect metadata from all samples
    LOGGER.info('Getting processed events')
    for proc in procs_decays:
        LOGGER.info(f'For proc {proc}')
        for sample in tqdm(processes[proc]):
            events[sample] = {}

            # Find all input files for this sample
            flist = find_sample_files(inDir, sample)
            file_list[sample] = flist
            # Extract cross section and total processed event count for luminosity scaling
            events[sample]['cross-section'] = getMetaInfo(sample, rmww=False)
            events[sample]['eventsProcessed'] = get_processed(flist)

    LOGGER.info('Getting cuts from DataFrame')
    for proc in procs_decays:
        LOGGER.info(f'For proc {proc}')
        for sample in tqdm(processes[proc]):
            flist = file_list.get(sample, [])
            has_file = bool(flist)
            processed = events[sample]['eventsProcessed']
            xsec = events[sample]['cross-section']

            # Compute luminosity scaling factor: lumi [ab^-1] * 1e6 [fb/ab] * xsec [fb] / N_events
            scale_sample = (lumi * 1e6 * xsec /
                            processed) if (processed and scaled) else 1.0

            # Process events from files if available
            if has_file and is_there_events(sample, inDir):
                for sel in sels:
                    if sel not in cuts:
                        continue
                    # Initialize cut statistics storage
                    events[sample][sel] = {'raw_count': {},
                                           'cut': {}, 'err': {}, 'filter': {}}
                    raw_counts = {cut: 0.0 for cut in cuts[sel].keys()}

                    # Process each event file
                    for f in flist:
                        df = get_df(f, branches) if branches else get_df(f)
                        if df is None or df.empty:
                            # Store filter expressions even if no data
                            for cut, filter in cuts[sel].items():
                                LOGGER.debug(f'{cut = }, {filter = }')
                                events[sample][sel]['filter'][cut] = filter
                            continue

                        # Apply cuts sequentially, accumulating event counts
                        df_mask = np.ones(len(df), dtype=bool)
                        for cut, filter in cuts[sel].items():
                            events[sample][sel]['filter'][cut] = filter
                            LOGGER.debug(f'{cut = }, {filter = }')
                            # Evaluate cut expression and accumulate passing events
                            count, df_mask = get_count(
                                df, df_mask, [f],
                                cut, filter,
                            )
                            raw_counts[cut] += float(count)

                        # Scale yields and compute Poisson errors
                        for cut, total_raw in raw_counts.items():
                            scale = scale_sample if scaled else 1.0
                            events[sample][sel]['raw_count'][cut] = float(
                                total_raw)
                            events[sample][sel]['cut'][cut] = float(
                                total_raw) * scale
                            events[sample][sel]['err'][cut] = np.sqrt(
                                total_raw) * scale

            else:
                # If no files available, use only generator-level first cut
                for sel in sels:
                    if sel not in cuts:
                        continue
                    events[sample][sel] = {'raw_count': {},
                                           'cut': {}, 'err': {}, 'filter': {}}
                    for cut, filter in cuts[sel].items():
                        events[sample][sel]['filter'][cut] = filter
                        LOGGER.debug(f'{cut = }, {filter = }')
                        # Only populate first cut with scaled generator events
                        if cut == 'cut0' and processed:
                            scale = lumi * 1e6 * xsec / processed
                            events[sample][sel]['raw_count'][cut] = processed
                            events[sample][sel]['cut'][cut] = scale * processed
                            events[sample][sel]['err'][cut] = scale * \
                                np.sqrt(processed)
                        else:
                            events[sample][sel]['raw_count'][cut] = 0
                            events[sample][sel]['cut'][cut] = 0
                            events[sample][sel]['err'][cut] = 0

    # Save event counts to JSON for bookkeeping
    LOGGER.info('Dumping events in a json file')
    out_json = f'{loc_json}/{ecm}/{cat}'
    dump_json(events, out_json, 'events')

    # Generate plots and tables for each selection
    for sel in sels:
        if sel not in cuts:
            continue
        if not tot:
            # Single-channel plots
            LOGGER.info('Preparing dictionary for cutflow plots')
            # Construct histograms from cut yields
            flow, flow_decay = get_flows(
                procs, processes,
                cuts, events,
                cat, sel,
                z_decays, H_decays,
                ecm=ecm, json_file=json_file,
                loc_json=out_json+f'/{sel}'
            )
            # Render main cutflow with signal, backgrounds, and significance
            LOGGER.info('Making Cutflow plot')
            CutFlow(
                flow, outDir, cat, sel,
                procs, colors, legend, cuts, cuts_label,
                ecm=ecm, lumi=lumi,
                outName='cutFlow', format=format,
                sig_scale=sig_scale
            )
            # Render per-decay efficiency plots
            LOGGER.info('Making CutflowDecays plot')
            CutFlowDecays(
                flow_decay, outDir, cat, sel,
                H_decays, cuts, cuts_label,
                ecm=ecm, lumi=lumi,
                format=format
            )
        else:
            # Compute both channel-specific and combined plots
            procs_cat, procs_tot = copy.deepcopy(procs), copy.deepcopy(procs)
            procs_cat[0] = f'Z{cat}H'

            # Channel-specific cutflow
            LOGGER.info('Preparing dictionary for cutflow plots')
            flow, flow_decay = get_flows(
                procs_cat, processes,
                cuts, events,
                cat, sel,
                z_decays, H_decays,
                ecm=ecm,
                json_file=json_file,
                loc_json=out_json+f'/{sel}'
            )
            LOGGER.info('Making Cutflow plot')
            CutFlow(
                flow, outDir, cat, sel,
                procs_cat, colors, legend, cuts, cuts_label,
                ecm=ecm, lumi=lumi, outName='cutFlow',
                format=format, sig_scale=sig_scale
            )
            LOGGER.info('Makinf CutflowDecays plot')
            CutFlowDecays(
                flow_decay, outDir, cat, sel,
                H_decays, cuts, cuts_label,
                ecm=ecm, lumi=lumi,
                format=format
            )

            # Combined (ee+mumu) cutflow
            flow_tot, flow_decay_tot = get_flows(
                procs_tot, processes,
                cuts, events, cat, sel,
                z_decays, H_decays,
                ecm=ecm, json_file=json_file,
                loc_json=loc_json+f'/{sel}',
                tot=True
            )
            LOGGER.info('Making Cutflow plot')
            CutFlow(
                flow_tot, outDir, cat, sel,
                procs_tot, colors, legend, cuts, cuts_label,
                ecm=ecm, lumi=lumi,
                suffix='_tot', format=format,
                sig_scale=sig_scale
            )
            LOGGER.info('Making CutflowDecays plot')
            CutFlowDecays(
                flow_decay_tot, outDir, cat, sel,
                H_decays, cuts, cuts_label,
                ecm=ecm, lumi=lumi,
                format=format, suffix='_tot',
                yMin=0 if cat == 'qq' else -30, yMax=160,
                tot=True
            )
