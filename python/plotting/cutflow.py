'''Cutflow event counting and plotting workflow.'''

################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, json, copy, uproot

import numpy as np

from tqdm import tqdm
from typing import Any
from functools import lru_cache



##########################################
### LOGGING AND CUSTOM MODULES IMPORTS ###
##########################################

from tools.process import getMetaInfo
from tools.utils import get_df
from logger import get_logger

LOGGER = get_logger(__name__)



########################
### HELPER FUNCTIONS ###
########################

@lru_cache(maxsize=None)
def _key_from_file(filepath: str, key: str) -> int:
    f = uproot.open(filepath)
    try:
        return int(f[key].value)
    except Exception:
        try:
            return int(f[key].array(entry_stop=1)[0])
        except Exception:
            LOGGER.warning(f"Couldn't find {key} in {filepath}, returning 0")
            return 0


def _get_bin_index(cut_name: str) -> int:
    try:
        return int(cut_name[3:]) + 1 if cut_name.startswith('cut') else 0
    except (ValueError, IndexError):
        return 0


@lru_cache(maxsize=None)
def _cut_from_file(filepath: str, cut_name: str) -> int:
    try:
        index = _get_bin_index(cut_name)
        if index <= 0:
            return 0
        root_file = uproot.open(filepath)
        for hist_type in ('TH1D', 'TH1F', 'TH1I', 'TH1S', 'TH1C'):
            try:
                return int(root_file[f'custom_objects/{hist_type}/cutFlow'].values()[index - 1])
            except (KeyError, IndexError, TypeError):
                continue
    except Exception:
        pass
    return 0


@lru_cache(maxsize=None)
def _col_from_file(filepath: str) -> set[str]:
    try:
        root_file = uproot.open(filepath)
        if 'events' in root_file:
            return set(map(str, root_file['events'].keys()))
    except Exception:
        pass
    return set()


def find_sample_files(inDir: str, sample: str) -> list[str]:
    path = os.path.abspath(os.path.join(inDir, sample))
    if os.path.isdir(path):
        return [os.path.join(path, name) for name in os.listdir(path)
                if name.endswith('.root') and os.path.isfile(os.path.join(path, name))]
    return [path + '.root'] if os.path.isfile(path + '.root') else []


def get_processed(files: list[str] | str, arg: str = 'eventsProcessed') -> int:
    files = [files] if isinstance(files, str) else files
    return int(sum(_key_from_file(os.fspath(filepath), arg) for filepath in files))


def get_cut(files: list[str] | str, cut_name: str) -> int:
    files = [files] if isinstance(files, str) else files
    return int(sum(_cut_from_file(os.fspath(filepath), cut_name) for filepath in files))


def getcut(
    df: Any,
    filter: str,
    mask: 'np.ndarray' | None = None,
) -> tuple[int, 'np.ndarray' | None]:
    if df is None or df.empty:
        return 0, mask
    selected = np.asarray(df.eval(filter), dtype=bool)
    mask = selected if mask is None else mask & selected
    return int(mask.sum()), mask


def get_count(
    df: Any,
    df_mask: 'np.ndarray' | None,
    file_list: list[str],
    cut_name: str,
    filter_expr: str,
) -> tuple[int, 'np.ndarray' | None]:
    if not filter_expr:
        return get_cut(file_list, cut_name), df_mask
    if df is None or df.empty:
        return 0, df_mask
    try:
        return getcut(df, filter_expr, df_mask)
    except Exception:
        LOGGER.warning("Couldn't compute the count for this cut")
        return 0, df_mask


def get_cuts(files: list[str] | str, cut_names: list[str]) -> dict[str, int]:
    files = [files] if isinstance(files, str) else files
    totals = {cut: 0 for cut in cut_names}
    for filepath in files:
        try:
            with uproot.open(os.fspath(filepath)) as root_file:
                histogram = next((root_file[f'custom_objects/{kind}/cutFlow']
                                  for kind in ('TH1D', 'TH1F', 'TH1I', 'TH1S', 'TH1C')
                                  if f'custom_objects/{kind}/cutFlow' in root_file), None)
                if histogram is None: continue
                values = histogram.values()
                for cut in cut_names:
                    index = _get_bin_index(cut) - 1
                    if 0 <= index < len(values):
                        totals[cut] += int(values[index])
        except Exception:
            continue
    return totals


def dump_json(flow: dict, outDir: str, outName: str) -> None:
    os.makedirs(outDir, exist_ok=True)
    with open(f'{outDir}/{outName}.json', 'w') as output:
        json.dump(copy.deepcopy(flow), output, indent=4)


def get_flows(
    events: dict,
    plots: dict[str, dict[str, list[str]]],
    cuts: dict[str, dict[str, str]],
    sel: str,
    json_file: bool = False,
    tot: bool = False,
    loc_json: str = '',
) -> tuple[dict, dict]:
    cut_names = list(cuts[sel])

    def aggregate(groups: dict[str, list[str]], skip_hinv: bool) -> dict:
        flow = {}
        for label, samples in groups.items():
            selected = [sample for sample in samples
                        if not skip_hinv or 'Hinv' not in sample]
            flow[label] = {
                'cut': {cut: sum(events[sample][sel]['cut'][cut] for sample in selected)
                        for cut in cut_names},
                'err': {cut: sum(events[sample][sel]['err'][cut] ** 2
                                for sample in selected) ** 0.5
                        for cut in cut_names},
            }
        return flow

    process_groups = {
        label: samples
        for group in ('signals', 'backgrounds')
        for label, samples in plots.get(group, {}).items()
    }
    flow = aggregate(process_groups, True)
    flow_decay = aggregate(plots.get('decays', {}), False)
    if json_file:
        suffix = '_tot' if tot else ''
        dump_json(flow, loc_json, f'flow_procs_{sel}{suffix}')
        dump_json(flow_decay, loc_json, f'flow_decay_{sel}{suffix}')
    return flow, flow_decay


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



#####################
### MAIN FUNCTION ###
#####################

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
    file_formats: list[str] = ['png'],
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

    LOGGER.info('Making cutflow plots')
    for sel in selected_sels:
        plot_config = {**procs_plots, 'decays': decay_plots.get('cat', {})}
        flow_procs, flow_decay = get_flows(events, plot_config, cuts, sel, json_file, False, out_json)
        CutFlow(flow_procs, outDir, sel, procs_plots, colors, legend, labels,
                ecm=ecm, lumi=lumi, file_formats=file_formats, sig_scale=sig_scale)
        CutFlowDecays(flow_decay, outDir, sel, labels, ecm=ecm, lumi=lumi, file_formats=file_formats, yMin=0, yMax=150)
        if tot:
            plot_config = {**procs_plots, 'decays': decay_plots.get('tot', {})}
            flow_procs, flow_decay = get_flows(events, plot_config, cuts, sel, json_file, True, out_json)
            CutFlowDecays(flow_decay, outDir, sel, labels, ecm=ecm, lumi=lumi, file_formats=file_formats,
                            yMin=0 if cat=='qq' else -50, yMax=160, tot=True)
