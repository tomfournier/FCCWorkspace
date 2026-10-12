'''Cutflow event counting and plotting workflow.'''

from typing import Any

import numpy as np
from tqdm import tqdm

from tools.process import getMetaInfo
from tools.utils import get_df
from logger import get_logger

LOGGER = get_logger(__name__)


def branches_from_cuts(
    cuts: dict[str, dict[str, str]],
    variables: list,
) -> list:
    '''Return variables referenced by any cut expression.'''
    import re

    return sorted({variable for selection in cuts.values()
                   for expression in selection.values() if expression
                   for variable in variables
                   if re.search(r'\b' + re.escape(variable) + r'\b', expression)})


def _required_branches(
    cuts: dict[str, dict[str, str]],
    selections: list[str],
    available: set[str],
    requested: list[str],
) -> list[str]:
    '''Use explicit branches or derive the minimal branch list from cuts.'''
    if requested:
        return list(dict.fromkeys(requested))
    return branches_from_cuts({selection: cuts[selection] for selection in selections},
                              sorted(available))


def get_cutflow(
    inDir: str,
    outDir: str,
    cat: str,
    sels: list[str],
    plots: dict[str, dict],
    colors: dict[str, dict[str, str]],
    legend: dict[str, dict[str, str]],
    cuts: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    format: list[str] = ['png'],
    ecm: int = 240,
    lumi: float = 10.8,
    sig_scale: float = 1.0,
    branches: list[str] = [],
    scaled: bool = True,
    tot: bool = False,
    json_file: bool = False,
    loc_json: str = '',
) -> None:
    '''Count events and render category and optional total cutflows.'''
    from plotting.python.helper import (
        _col_from_file, dump_json, get_count,
        get_cuts, get_flows, get_processed, find_sample_files,
    )
    from plotting.plotting import CutFlow, CutFlowDecays

    procs_plots = {k:plots[k] for k in ('signals', 'backgrounds')}
    decay_plots = plots['decays']

    active_decay_groups = [decay_plots.get('cat', {})]
    if tot:
        active_decay_groups.append(decay_plots.get('tot', {}))
    samples = sorted({sample
                      for group in [*procs_plots.values(), *active_decay_groups]
                      for sample_list in group.values()
                      for sample in sample_list})

    # Initialize event and file list dictionaries.
    events: dict[str, dict[str, float | int | str | dict[str, float | str | dict[str, Any]]]] = {}
    file_list: dict[str, list[str]] = {}

    # Collect metadata from all samples
    LOGGER.info('Getting processed events')
    for sample in tqdm(samples):
        events[sample] = {}

        # Find all input files for the sample
        files = find_sample_files(inDir, sample)
        file_list[sample] = files

        # Extract cross section and total processed event count for luminosity scaling
        events[sample] = {'cross-section': getMetaInfo(sample, rmww=True),
                          'eventsProcessed': get_processed(files)}

    selected_sels = [sel for sel in sels if sel in cuts]
    LOGGER.info('Getting cuts from DataFrame')
    for sample in tqdm(samples):
        flist = file_list.get(sample, [])
        processed = events[sample]['eventsProcessed']
        xsec  = events[sample]['cross-section']
        scale = (lumi * 1e6 * xsec / processed if processed and scaled else 1.0)

        for sel in selected_sels:
            events[sample][sel] = {'raw_count': {cut: 0. for cut in cuts[sel]},
                                   'cut': {}, 'err': {}, 'filter': dict(cuts[sel]),
                                   'label': dict(labels[sel])}

        event_files = [f for f in flist if _col_from_file(f)]
        histogram_sels = [sel for sel in selected_sels if all(not expr for expr in cuts[sel].values())]
        for sel in histogram_sels:
            events[sample][sel]['raw_count'].update(get_cuts(flist, list(cuts[sel])))

        data_sels = [sel for sel in selected_sels if sel not in histogram_sels]
        if event_files and data_sels:
            available = set().union(*(_col_from_file(path) for path in event_files))
            read_branches = _required_branches(cuts, data_sels, available, branches)
            for filepath in event_files:
                df = get_df(filepath, read_branches)
                if df is None or df.empty: continue
                for sel in data_sels:
                    mask = np.ones(len(df), dtype=bool)
                    empty_cuts = [cut for cut, expr in cuts[sel].items() if not expr]
                    histogram_counts = get_cuts([filepath], empty_cuts)
                    for cut, expr in cuts[sel].items():
                        if expr:
                            count, mask = get_count(df, mask, [], cut, expr)
                        else:
                            count = histogram_counts[cut]
                        events[sample][sel]['raw_count'][cut] += count

        for sel in selected_sels:
            raw_counts = events[sample][sel]['raw_count']
            if not event_files and processed:
                raw_counts['cut0'] = float(processed)
            events[sample][sel]['cut'] = {cut: count          * scale for cut, count in raw_counts.items()}
            events[sample][sel]['err'] = {cut: np.sqrt(count) * scale for cut, count in raw_counts.items()}

    # Save event counts to JSON for bookkeeping
    LOGGER.info('Dumping events in a json file')
    out_json = f'{loc_json}/{ecm}/{cat}'
    dump_json(events, out_json, 'events')

    for sel in selected_sels:
        plot_config = {**procs_plots, 'decays': decay_plots.get('cat', {})}
        flow_procs, flow_decay = get_flows(events, plot_config, cuts, sel, json_file, False, out_json)
        CutFlow(flow_procs, outDir, cat, sel, procs_plots, colors, legend, labels,
                ecm=ecm, lumi=lumi, format=format, sig_scale=sig_scale)
        CutFlowDecays(flow_decay, outDir, cat, sel, labels, ecm=ecm, lumi=lumi, format=format, yMin=0, yMax=150)
        if tot:
            plot_config = {**procs_plots, 'decays': decay_plots.get('tot', {})}
            flow_procs, flow_decay = get_flows(events, plot_config, cuts, sel, json_file, True, out_json)
            CutFlowDecays(flow_decay, outDir, cat, sel, labels, ecm=ecm, lumi=lumi, format=format,
                            yMin=0 if cat=='qq' else -30, yMax=160, tot=True)
