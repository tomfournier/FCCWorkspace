import os, time
t = time.time()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)

parser.add_argument('--baseline', help='Baseline selection', action='store_true')
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
parser.add_argument('--vis', help='Add E_vis > 10 GeV cut', action='store_true')
parser.add_argument('--all', help='Run 2-BDT for all available selections', action='store_true')

parser.add_argument('--process', help='Run only process_input.py', action='store_true')
parser.add_argument('--train', help='Run only train_bdt.py', action='store_true')
parser.add_argument('--eval', help='Run only evaluation.py', action='store_true')
parser.add_argument('--full', help='Run all files in 2-BDT', action='store_true')
arg = parser.parse_args()

cats, ecm = [arg.cat] if arg.cat=='ee' or arg.cat=='mumu' else ['mumu', 'ee'], '--ecm 365' if arg.ecm==365 else ''
sels = []

if not arg.all and not arg.baseline and not arg.recoil120 and not arg.miss and not arg.bdt and not arg.vis:
    print('\n-------------------------------------------------------------\n')
    print('No selection was selected, please select one to run this code')
    print('\n-------------------------------------------------------------\n')
    exit(0)

if not arg.full and not arg.process and not arg.train and not arg.eval:
    print('\n--------------------------------------------------------\n')
    print('No file was selected, please select one to run this code')
    print('\n--------------------------------------------------------\n')
    exit(0)

if (arg.all and arg.recoil120) or (arg.all and arg.miss) or (arg.all and arg.bdt) or (arg.all and arg.vis):
    selection = input('You have put --all and another selection in argument when running this file\n' 
                      'Do you want to run all selection? [yes, no]: ')
    if selection=='no':
        arg.all = False

if arg.all:
    for i in [' ', '--recoil120', '--miss', '--bdt', '--vis']: sels.append(i)
else:
    if arg.baseline:  sels.append(' ')
    if arg.recoil120: sels.append('--recoil120')
    if arg.miss:      sels.append('--miss')
    if arg.bdt:       sels.append('--bdt')
    if arg.vis:       sels.append('--vis')

if (arg.full and arg.process) or (arg.full and arg.train) or (arg.full and arg.eval):
    file = input('You have put --full and another file to run in argument when running this file\n' 
                 'Do you want to run all files in 2-BDT? [yes, no]: ')
    if file=='no':
        arg.full = False

for sel in sels:
    for cat in cats:
        cmd = ''
        if arg.full:
            cmd += f'python 2-BDT/process_input.py --cat {cat} {ecm} {sel}\n'
            cmd += f'python 2-BDT/train_bdt.py --cat {cat} {ecm} {sel}\n'
            cmd += f'python 2-BDT/evaluation.py --cat {cat} {ecm} {sel}\n'
        else:
            if arg.process: cmd += f'python 2-BDT/process_input.py --cat {cat} {ecm} {sel}\n'
            if arg.train:   cmd += f'python 2-BDT/train_bdt.py --cat {cat} {ecm} {sel}\n'
            if arg.eval:    cmd += f'python 2-BDT/evaluation.py --cat {cat} {ecm} {sel}\n'
        
        if sel==' ': selection = 'Baseline'
        print(f'\nRunning the code for {cat} channel at ecm = {arg.ecm} GeV for the selection {sel}\n')
        os.system(cmd)

print('\n\n---------------------------------------------')
print('---------------------------------------------\n')
print(f'Time taken to run the full code: {time.time()-t:.1f} s')
print('\n---------------------------------------------')
print('---------------------------------------------\n\n')
