import json, time, argparse
import numpy as np

def extra_arg(args: list) -> str:
    cmd = ''
    for arg in args: cmd += f' --{arg}'
    return cmd

def argument(cat: bool = True, sel: bool = True, lumi: bool = False, comb: bool = False, 
             extra: bool = False, run: bool = False, ILC: bool = False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365, -1], type=int, default=240)
    
    if cat:
        parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')

    if lumi:
        parser.add_argument('--lumi', help='Integrated luminosity in attobarns', type=float, default=-1)

    if sel:
        parser.add_argument('--recoil', help='Cut with 120 GeV < recoil mass < 140 GeV',    action='store_true')
        parser.add_argument('--miss',   help='Add the cos(theta_miss) < 0.98 cut',          action='store_true')
        parser.add_argument('--bdt',    help='Add cos(theta_miss) cut as input to the BDT', action='store_true')
        parser.add_argument('--lead',   help='Add the p_leading and p_subleading cuts',     action='store_true')
        parser.add_argument('--vis',    help='Add E_vis > 100 GeV cut',                     action='store_true')
        parser.add_argument('--sep',    help='Separate events by using E_vis',              action='store_true')

    if comb:
        parser.add_argument("--combine", help='Combine the channel to do the fit', action='store_true')

    if extra:
        parser.add_argument("--target", type=str,   help="Target pseudodata",      default="bb")
        parser.add_argument("--pert",   type=float, help="Target pseudodata size", default=1.0)
        parser.add_argument("--run",                help="Run combine",            action='store_true')
        parser.add_argument("--freezeBackgrounds",  help="Freeze backgrounds",     action='store_true')
        parser.add_argument("--floatBackgrounds",   help="Float backgrounds",      action='store_true')
        parser.add_argument("--plot_dc",            help="Plot datacard",          action='store_true')

    if ILC:
        parser.add_argument("--polL", help="Scale to left polarization",  action='store_true')
        parser.add_argument("--polR", help="Scale to right polarization", action='store_true')
        parser.add_argument("--ILC",  help="Scale to ILC luminosity",     action='store_true')

    if run:
        parser.add_argument('--presel',  help='Run only pre-selection.py',     action='store_true')
        parser.add_argument('--final',   help='Run only final-selection.py',   action='store_true')
        parser.add_argument('--plots',   help='Run only plots.py',             action='store_true')

        parser.add_argument('--input',   help='Run only process_input.py',     action='store_true')
        parser.add_argument('--train',   help='Run only train_bdt.py',         action='store_true')
        parser.add_argument('--eval',    help='Run only evaluation.py',        action='store_true')

        parser.add_argument('--sel',     help='Run only selection.py',         action='store_true')
        parser.add_argument('--process', help='Run only process_histogram.py', action='store_true')
        parser.add_argument('--comb',    help='Run only combine.py',           action='store_true')

        parser.add_argument('--full',    help='Run all files in 2-BDT',        action='store_true')

        parser.add_argument('--multi',   help='Run combined selection, rely combined selection with "-"'
                            ' and separate the combined selections with "_"', type=list, default='')
            
    arg = parser.parse_args()
    args = [arg]
    
    sel_list   =          [arg.recoil, arg.miss, arg.bdt, arg.lead, arg.vis, arg.sep]
    extra_list = np.array([   'recoil',   'miss',   'bdt',   'lead',   'vis',   'sep'])
    if sel:   args.append(sel_list)
    if extra: args.append(extra_arg(extra_list[np.where(sel_list)]))
    return args

def dump_json(arg, file, indent=4):
    with open(file, mode='w', encoding='utf-8') as fOut:
        json.dump(arg, fOut, indent=indent)

def load_json(file):
    with open(file, mode='r', encoding='utf-8') as fIn:
        arg = json.load(fIn)
    return arg

def warning(cat, comb: bool = False, is_there_comb: bool = False):
    if not is_there_comb and cat=='':
        print('\n-----------------------------------------\n')
        print('Final state was not selected. Aborting...')
        print('\n-----------------------------------------\n')
        exit(0)

    if is_there_comb and cat=='' and not comb:
        print('\n-----------------------------------------------------\n')
        print('Final state or combine were not selected. Aborting...')
        print('\n-----------------------------------------------------\n')
        exit(0)

def timer(t):
    dt = time.time() - t
    h, m, s = dt//3600, dt//60, dt%60 

    print('\n\n-----------------------------------------\n')
    print(f'Time taken to run the code: {h:.0f} h {m:.0f} min {s:.2f} s')
    print('\n-----------------------------------------\n\n')
