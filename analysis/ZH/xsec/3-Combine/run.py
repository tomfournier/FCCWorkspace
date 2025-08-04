import os, time
t = time.time()
import numpy as np

from importlib import import_module
userConfig = import_module('userConfig')
from userConfig import loc, argument, extra_arg, dump_json, timer

arg = argument(lumi=True, run=True)
cats = [arg.cat] if arg.cat=='ee' or arg.cat=='mumu' else ['mumu', 'ee']
ecms, sels = [240, 365] if arg.ecm==-1 else [arg.ecm], []

ind = [arg.sel, arg.process, arg.plots, arg.comb]

if not arg.full and not np.any(ind):
    print('\n--------------------------------------------------------\n')
    print('No file was selected, please select one to run this code')
    print('\n--------------------------------------------------------\n')
    exit(0)

if arg.multi!='':
    combs = arg.multi.split('_')
    for comb in combs: sels.append(extra_arg(comb.split('-')))

if arg.full and np.any(ind):
    file = input('You have put --full and another file to run in argument when running this file\n' 
                 'Do you want to run all files in 2-BDT? [yes, no]: ')
    if file=='no': arg.full = False

for sel in sels:
    for ecm in ecms:
        if arg.lumi==-1: lumi = 10.8 if ecm==240 else 3.1
        else:            lumi = arg.lumi
        for cat in cats:
            cmd, arg = '', {'cat': cat, 'ecm': ecm, 'sel': sel, 'lumi':lumi}
            dump_json(f'{loc.JSON}/Combine_config.json', arg)

            if arg.full:
                cmd += f'python 3-Combine/pre-selection.py\n'
                cmd += f'python 3-Combine/final-selection.py\n'
                cmd += f'python 3-Combine/plots.py\n'
            else:
                if arg.sel:     cmd += f'python 3-Combine/selection.py\n'
                if arg.process: cmd += f'python 3-Combine/process_histogram.py\n'
                if arg.plots:   cmd += f'python 3-Combine/plots.py\n'
                if arg.comb:    cmd += f'python 3-Combine/combine.py\n'
            
            if sel==' ': selection = 'Baseline'
            print(f'\nRunning the code for {cat} channel at ecm = {arg.ecm} GeV for the selection {sel}\n')
            os.system(cmd)

timer(t)
