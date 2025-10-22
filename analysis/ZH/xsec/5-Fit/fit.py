import os, argparse, importlib
import numpy as np

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
parser.add_argument('--sel', help='Selection with which you fit the histograms', 
                    type=str, default='Baseline')

parser.add_argument("--pert",   help="Target pseudodata size", 
                    type=float, default=1.0)
parser.add_argument("--target", help="Target pseudodata", 
                    type=str, default="")

parser.add_argument("--bias",    help="Nominal fit or bias test", action='store_true')
parser.add_argument("--combine", help='Combine the channel to do the fit', action='store_true')
arg = parser.parse_args()

if arg.cat=='' and not arg.combine:
    print('\n----------------------------------------------------------------------------\n')
    print('Final state or combine were not selected, please select one to run this code')
    print('\n----------------------------------------------------------------------------\n')
    exit(0)
if arg.combine: arg.cat = 'combined'
args = [arg.cat, arg.ecm, arg.sel]

if not arg.bias:
    dir, dc = get_loc(loc.COMBINE_NOMINAL, *args), get_loc(loc.NOMINAL_DATACARD, *args)
    ws, log = get_loc(loc.NOMINAL_WS, *args),      get_loc(loc.NOMINAL_LOG, *args)
    res, tp = get_loc(loc.NOMINAL_RESULT, *args),  "nominal"
else:
    dir, dc = get_loc(loc.COMBINE_BIAS, *args),    get_loc(loc.BIAS_DATACARD, *args)
    ws, log = get_loc(loc.BIAS_WS, *args),         get_loc(loc.BIAS_LOG, *args)
    res, tp = get_loc(loc.BIAS_FIT_RESULT, *args), "bias"

comb, tar   = "_combined" if arg.combine else "", f"_{arg.target}" if arg.bias else ""
dc_comb = get_loc(loc.COMBINE, '', arg.ecm, arg.sel)

cmd = f"cd {dir};"
if arg.combine:
    dc_mu, dc_e = f"{dc_comb}/mumu/{tp}/datacard", f"{dc_comb}/ee/{tp}/datacard"
    cmd += f"combineCards.py {dc_mu}/datacard{tar}.txt {dc_e}/datacard{tar}.txt > {dc}/datacard{tar}{comb}.txt;"
cmd += f"text2workspace.py {dc}/datacard{tar}{comb}.txt -v 10 --X-allow-no-background -m 125 -o {ws}/ws{tar}.root &> {log}/log_text2workspace{tar}.txt;"
cmd += f"cd {ws};"
cmd += f"combine ws{tar}.root -M MultiDimFit -m 125 -v 10 -t 0 --expectSignal=1 -n Xsec &> {log}/log_results{tar}.txt"

print('----->[Info] Running the fit')
for i in [dc, ws, log, res]:
    if not os.path.isdir(i):
        os.system(f'mkdir -p {i}')
os.system(cmd)
print('----->[Info] Fit done, extracting results')

    

mu, err = np.inf, np.inf
with open(f"{log}/log_results{tar}.txt") as file:
    for l in file.readlines():
        plit = ' '.join(l.split("\t"))
        plit = ' '.join(plit.split("\n"))
        plit = plit.split(' ')

        if plit[-2]=='(limited)':
            if (plit[0]=="r") & (plit[3]=="=") & (plit[6]=="+/-"):
                mu, err = float(plit[4]), float(plit[8])
file.close()

if (mu==np.inf) & (err==np.inf):
    print("----->[Error] Couldn't extract value of the fit, go to the log to have more informations")
    exit(0)
else:
    print('----->[Info] Results successfully extracted')
    print(f'\tmu = {mu} +/- {err}')
    print(f'----->[Info] Uncertainty obtained on ZH cross-section: {err*100:.2f} %')
    f = np.array([mu, err])
    np.savetxt(f"{res}/results{tar}.txt", f)
    print(f'----->[Info] Saved result in {res}/results{tar}.txt')
    