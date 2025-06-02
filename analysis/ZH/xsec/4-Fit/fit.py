import os
import importlib
import numpy as np
import argparse

userConfig = importlib.import_module('userConfig')
from userConfig import loc, combine

parser = argparse.ArgumentParser()
parser.add_argument("--bias", help="Nominal fit or bias test", action='store_true')
parser.add_argument("--pert", type=float, help="Target pseudodata size", default=1.0)
parser.add_argument("--target", type=str, help="Target pseudodata", default="")
args = parser.parse_args()

if not args.bias:
    dir, dc, tp  = loc.COMBINE_NOMINAL, loc.NOMINAL_DATACARD, "nominal"
    ws, log, res = loc.NOMINAL_WS, loc.NOMINAL_LOG, loc.NOMINAL_RESULT
else:
    dir, dc, tp  = loc.COMBINE_BIAS, loc.BIAS_DATACARD, "bias"
    ws, log, res = loc.BIAS_WS, loc.BIAS_LOG, loc.BIAS_FIT_RESULT

comb, tar = "_combined" if combine else "", f"_{args.target}" if args.bias else ""

cmd = f"cd {dir};"
if combine:
    dc_mu, dc_e = f"{loc.COMBINE}/mumu/{tp}/datacard", f"{loc.COMBINE}/ee/{tp}/datacard"
    cmd += f"combineCards.py {dc_mu}/datacard{tar}.txt {dc_e}/datacard{tar}.txt > {dc}/datacard{tar}{comb}.txt;"
cmd += f"text2workspace.py {dc}/datacard{tar}{comb}.txt -v 10 --X-allow-no-background -m 125 -o {ws}/ws{tar}.root &> {log}/log_text2workspace.txt;"
cmd += f"cd {ws};"
cmd += f"combine ws{tar}.root -M MultiDimFit -m 125 -v 10 -t 0 --rMax {args.pert+0.1} --rMin {args.pert-0.1} --expectSignal={args.pert} -n Xsec &> {log}/log_results{tar}.txt"

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
    print(f'----->[Info] Uncertainty on cross-section: {err*100:.2f} %')
    f = np.array([mu, err])
    np.savetxt(f"{res}/results{tar}.txt", f)
    print(f'----->[Info] Saved result in {res}/results{tar}.txt')