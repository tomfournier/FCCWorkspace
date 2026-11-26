#!/usr/bin/env python3
"""
Simplified 1D likelihood scan plotter using matplotlib and uproot.
Optimized for speed and simplicity while maintaining plot1DScan.py structure.
"""

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

import time

from pathlib import Path

from zh_xsec.func.self_coupling import get_params, coeffs_label

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log, parse_args
from zh_xsec.parsing import create_parser
from logger import get_logger
parser = create_parser('6-Self-coupling', 'plots')
arg = parse_args(parser, False, False)
set_log(arg)

LOGGER = get_logger(__name__)



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from zh_xsec.userConfig import loc
loc.set_default_type(Path)
from tools.utils import timer
from fitting import plot_1d_scans, plot_2d_scans, params_label
from matplotlib import rcParams

rcParams['text.latex.preamble'] += r'\usepackage{amssymb}'



####################
### CONFIG SETUP ###
####################

# Parse main inputs
sels: list[str] = arg.sels.split('-')
cats: list[str] = [cat for cat in arg.cat.split('-') if cat]
if arg.lep:
    cats = cats + ['leptonic'] if cats else ['leptonic']
if arg.combine:
    cats = cats + ['combined'] if cats else ['combined']
params: list[str] = arg.param.split('-')
model_params = get_params(arg.model)
if any(p not in model_params for p in params):
    raise ValueError(f'At least one parameter in {params} is not defined in {arg.model}, '
                     'choose a model that contains all parameters')


##########################
### EXECUTION FUNCTION ###
##########################

def main():

    colors  = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    filename = f'higgsCombineXsec_{"_".join(model_params)}.MultiDimFit.mH125.root'

    # Individual plots for all combinations
    if arg.which == '':
        for cat, sel in [(c, s) for c in cats for s in sels]:
            inputdir  = loc.get('SELF_WS',     cat, '', sel)
            outputdir = loc.get('SELF_RESULT', cat, '', sel)
            scan = (inputdir / filename, "Observed", next(iter(colors)))
            for param in params:
                other_params = [p for p in params if p!=param] if len(params)>1 else []
                plot_1d_scans([scan], outputdir, param, arg.y_cut, arg.y_max,
                              sig2=arg.sig2, suffix='_'+param, profile=arg.do_profile, other_params=other_params, coeffs_label=coeffs_label)
            if len(params) > 1:
                scan = (inputdir / filename, 'Best fit', 'black')
                from itertools import combinations
                for x_param, y_param in list(combinations(params, 2)):
                    plot_2d_scans([scan], outputdir, x_param, y_param, arg.y_cut, sig2=arg.sig2, coeffs_label=coeffs_label)
    else:
        # Comparison plot: configure varying parameter and fixed values
        from itertools import product

        config = {'cat': (cats, [[''], sels]), 'sel': (sels, [cats, ['']])}
        varying, fixed_lists = config[arg.which]
        insert_pos = ['cat', '', 'sel'].index(arg.which)

        for fixed_vals in product(*fixed_lists):
            fixed_str = '_'.join(str(f) for f in fixed_vals)
            param_label = ', '.join(params_label.get(str(f), f) for f in fixed_vals)

            all_scans = []
            for i, var_val in enumerate(varying):
                param = list(fixed_vals).insert(insert_pos, var_val)
                all_scans.append((loc.get('SELF_WS', *param) / filename, str(var_val), colors[i % len(colors)]))

            out = loc.get('PLOT_SCAN') / arg.which
            out.mkdir(exist_ok=True, parents=True)
            for param in params:
                other_params = [p for p in params if p!=param] if len(params)>1 else []
                plot_1d_scans(all_scans, out, param, arg.y_cut, arg.y_max,
                              fixed_str+'_'+param, param_label, arg.sig2, arg.do_profile, other_params, coeffs_label=coeffs_label)
            LOGGER.info(param)
            if len(params) > 1:
                from itertools import combinations
                for x_param, y_param in list(combinations(params, 2)):
                    for scan in all_scans:
                        plot_2d_scans([scan], outputdir, x_param, y_param, arg.y_max, sig2=arg.sig2, coeffs_label=coeffs_label)


##########################
### EXECUTION FUNCTION ###
##########################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        if arg.timer: timer(t)
