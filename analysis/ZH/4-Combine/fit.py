import os
import importlib
import numpy as np
import pandas as pd

userConfig = importlib.import_module('userConfig')
outputdir = userConfig.loc.COMBINE

cmd = f"cd {outputdir};"
cmd += "text2workspace.py datacard.txt -v 10 --X-allow-no-background -m 125 -o ws.root &> log/log_text2workspace.txt;"
# cmd += "combine -M MultiDimFit -v 5 --rMin 0.9 --rMax 1.1 --setParameters r=1 ws.root"
cmd += "combine ws.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n Xsec &> log/log_results.txt"
# cmd += "combine ws.root -o fit_output.root -t -0 --expectSignal=1 --binByBinStat"

print('----->[Info] Running the fit')
if not os.path.isdir(f'{outputdir}/log'):
    os.system(f'mkdir {outputdir}/log')
os.system(cmd)
print('----->[Info] Fit done, extracting results')

mu, err = np.inf, np.inf
with open(f"{outputdir}/log/log_results.txt") as file:
    for l in file.readlines():
        plit = ' '.join(l.split("\t"))
        plit = ' '.join(plit.split("\n"))
        plit = plit.split(' ')

        if plit[-2]=='(limited)':
            if (plit[0]=="r") & (plit[3]=="=") & (plit[6]=="+/-"):
                mu, err = float(plit[4]), float(plit[8])
file.close()

if (mu==np.inf) & (err==np.inf):
    print("----->[Error] Couldn't extract value of the fit, Aborting...")
    exit(0)
else:
    print('----->[Info] Results successfully extracted')
    print(f'\tmu = {mu} +/- {err}')
    df = pd.Series([mu, err], index=['mu', 'err'])
    df.to_csv(f'{outputdir}/results.csv')
    print(f'----->[Info] Saved result in {outputdir}/log/results.csv')