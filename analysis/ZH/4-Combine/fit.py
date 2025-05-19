import os
import importlib
import numpy as np
import pandas as pd

userConfig = importlib.import_module('userConfig')
outputdir = userConfig.loc.COMBINE


if not userConfig.combine:
    cmd = f"cd {outputdir};"
    cmd += "text2workspace.py datacard.txt -v 10 --X-allow-no-background -m 125 -o ws.root &> log/log_text2workspace.txt;"
    cmd += "combine ws.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n Xsec &> log/log_results.txt"

    File = f"{outputdir}/log/log_results.txt"

    print('----->[Info] Running the fit')
    if not os.path.isdir(f'{outputdir}/log'):
        os.system(f'mkdir {outputdir}/log')
    os.system(cmd)
    print('----->[Info] Fit done, extracting results')

    Fout = f'{outputdir}/log/results.csv'

else:
    cmd = f"cd {outputdir}/combined;"
    cmd += f"combineCards.py ../mumu/datacard.txt ../ee/datacard.txt > datacard_combined.txt;"
    cmd += "text2workspace.py datacard_combined.txt -v 10 --X-allow-no-background -m 125 -o ws.root &> log/log_text2workspace.txt;"
    cmd += "combine ws.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n Xsec &> log/log_results.txt"
    
    File = f"{outputdir}/combined/log/log_results.txt"

    print('----->[Info] Running the fit')
    if not os.path.isdir(f'{outputdir}/combined'):
        os.system(f'mkdir {outputdir}/combined')
    
    if not os.path.isdir(f'{outputdir}/combined/log'):
        os.system(f'mkdir {outputdir}/combined/log')
    os.system(cmd)
    print('----->[Info] Fit done, extracting results')

    Fout = f'{outputdir}/combined/log/results.csv'


mu, err = np.inf, np.inf
with open(File) as file:
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
    df.to_csv(Fout)
    print(f'----->[Info] Saved result in {Fout}')