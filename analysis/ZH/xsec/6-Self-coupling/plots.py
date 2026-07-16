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

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,
    cat_default='',
    allow_empty=True,
    no_ecm=True,
    include_sels=True,
    fit_plot=True,
    description='Fit Plots Script'
)
arg = parse_args(parser, False, False)
set_log(arg)

LOGGER = get_logger(__name__)



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from package.userConfig import loc
loc.set_default_type(Path)
from package.config import timer
from package.func.fit import plot_1d_scans, plot_2d_scans, params_label



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



##########################
### EXECUTION FUNCTION ###
##########################

def main():

    colors  = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Individual plots for all combinations
    if arg.which == '':
        for cat, sel in [(c, s) for c in cats for s in sels]:
            inDir  = loc.get('NLO_WS',     cat, '', sel)
            outDir = loc.get('NLO_RESULT', cat, '', sel)
            fIn = 'higgsCombineXsec.MultiDimFit.mH125.123456.root' if arg.toy \
                else 'higgsCombineXsec.MultiDimFit.mH125.root'
            scan = (inDir / fIn, "Observed", colors[0])
            for param in params:
                other_params = [p for p in params if p!=param] if len(params)>1 else []
                plot_1d_scans([scan], outDir, param,
                              arg.y_cut, arg.y_max,
                              sig2=arg.sig2, suffix='_'+param, other_params=other_params)
            if len(params) > 1:
                scan = (inDir / fIn, 'Best fit', 'black')
                from itertools import combinations
                for x, y in list(combinations(params, 2)):
                    plot_2d_scans([scan], outDir, x, y, arg.y_cut,
                                  sig2=arg.sig2)
    else:
        # Comparison plot: configure varying parameter and fixed values
        from itertools import product

        config = {
                'cat': (cats, [[''], sels]),
                'sel': (sels, [cats, ['']])
        }
        varying, fixed_lists = config[arg.which]
        insert_pos = ['cat', '', 'sel'].index(arg.which)

        for fixed_vals in product(*fixed_lists):
            fixed_str = '_'.join(str(f) for f in fixed_vals)
            param_label = ', '.join(params_label.get(str(f), f) for f in fixed_vals)

            all_scans = []
            for i, var_val in enumerate(varying):
                param = list(fixed_vals)
                param.insert(insert_pos, var_val)

                inDir = loc.get('NLO_WS', *param)
                fIn = 'higgsCombineXsec.MultiDimFit.mH125.123456.root' if arg.toy \
                    else 'higgsCombineXsec.MultiDimFit.mH125.root'
                scan  = (inDir / fIn,
                         str(var_val),
                         colors[i % len(colors)])
                all_scans.append(scan)

            out: Path = loc.get('PLOTS_FIT_SCAN') / arg.which
            out.mkdir(exist_ok=True, parents=True)
            for param in params:
                other_params = [p for p in params if p!=param] if len(params)>1 else []
                plot_1d_scans(
                    all_scans, out, param,
                    arg.y_cut, arg.y_max,
                    fixed_str+'_'+param, param_label,
                    arg.sig2, other_params=other_params
                )
            LOGGER.info(param)
            if len(params) > 1:
                from itertools import combinations
                for x, y in list(combinations(params, 2)):
                    for scan in all_scans:
                        plot_2d_scans([scan], outDir, x, y, arg.y_cut,
                                      z_cut=arg.y_max, sig2=arg.sig2)


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
