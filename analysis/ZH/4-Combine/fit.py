import os
import importlib
import numpy as np
import pandas as pd
import argparse

userConfig = importlib.import_module('userConfig')

parser = argparse.ArgumentParser()
parser.add_argument("--outputdir", type=str, help="Output directory", default=userConfig.loc.COMBINE)
parser.add_argument("--target", type=str, help="Target pseudodata", default="")
args = parser.parse_args()

outputdir = args.outputdir

if not userConfig.combine:
    cmd = f"cd {outputdir};"
    if args.target=="":
        cmd += "text2workspace.py datacard.txt -v 10 --X-allow-no-background -m 125 -o ws.root &> log/log_text2workspace.txt;"
        cmd += "combine ws.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n Xsec &> log/log_results.txt"
        File = f"{outputdir}/log/log_results.txt"

        print('----->[Info] Running the fit')
        if not os.path.isdir(f'{outputdir}/log'):
            os.system(f'mkdir -p {outputdir}/log')
        os.system(cmd)
        print('----->[Info] Fit done, extracting results')

        Fout = f'{outputdir}/results/results.txt'

    else:
        cmd += f"text2workspace.py datacard/datacard_{args.target}.txt -v 10 --X-allow-no-background -m 125 -o datacard/ws_{args.target}.root &> log/log_text2workspace_{args.target}.txt;"
        cmd += f"combine datacard/ws_{args.target}.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n Xsec &> log/log_results_{args.target}.txt"
        File = f"{outputdir}/log/log_results_{args.target}.txt"

        print('----->[Info] Running the fit')
        if not os.path.isdir(f'{outputdir}/log'):
            os.system(f'mkdir -p {outputdir}/log')
        os.system(cmd)
        print('----->[Info] Fit done, extracting results')

        Fout = f'{outputdir}/results/results_{args.target}.txt'

else:
    cmd = f"cd {outputdir};"
    if args.target=="":
        cmd += "text2workspace.py datacard.txt -v 10 --X-allow-no-background -m 125 -o ws.root &> log/log_text2workspace.txt;"
        cmd += f"combineCards.py ../mumu/datacard.txt ../ee/datacard.txt > datacard_combined.txt;"
        cmd += "combine ws.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n Xsec &> log/log_results.txt"
        File = f"{outputdir}/log/log_results.txt"

        print('----->[Info] Running the fit')        
        if not os.path.isdir(f'{outputdir}/combined/log'):
            os.system(f'mkdir -p {outputdir}/combined/log')
        os.system(cmd)
        print('----->[Info] Fit done, extracting results')

        Fout = f'{outputdir}/combined/results/results.txt'

    else:
        cmd += f"text2workspace.py datacard/datacard_{args.target}.txt -v 10 --X-allow-no-background -m 125 -o datacard/ws_{args.target}.root &> log/log_text2workspace_{args.target}.txt;"
        cmd += f"combineCards.py ../mumu/datacard_{args.target}.txt ../ee/datacard_{args.target}.txt > datacard_combined_{args.target}.txt;"
        cmd += f"combine datacard/ws_{args.target}.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n Xsec &> log/log_results_{args.target}.txt"
        File = f"{outputdir}/log/log_results_{args.target}.txt"

        print('----->[Info] Running the fit')        
        if not os.path.isdir(f'{outputdir}/combined/log'):
            os.system(f'mkdir -p {outputdir}/combined/log')
        os.system(cmd)
        print('----->[Info] Fit done, extracting results')

        Fout = f'{outputdir}/results/results_{args.target}.txt'
    

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
    if not os.path.isdir(f'{outputdir}/results'):
        os.system(f'mkdir -p {outputdir}/results')
    print('----->[Info] Results successfully extracted')
    print(f'\tmu = {mu} +/- {err}')
    f = np.array([mu, err])
    np.savetxt(Fout, f)
    print(f'----->[Info] Saved result in {Fout}')