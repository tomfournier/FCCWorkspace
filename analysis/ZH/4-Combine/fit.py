import os
import importlib
# import ROOT

userConfig = importlib.import_module('userConfig')
outputdir = userConfig.loc.COMBINE

cmd = f"cd analysis/ZH/output/combine/mumu;"
cmd += "text2workspace.py datacard.txt -v 10 --X-allow-no-background -o ws.root;"
# cmd += "combine -M MultiDimFit -v 5 --rMin 0.9 --rMax 1.1 --setParameters r=1 ws.root"
cmd += "combine ws.root -M MultiDimFit -m 125 -v 10 -t -1 --expectSignal=1 -n xsec"
# cmd += "combine ws.root -o fit_output.root -t -0 --expectSignal=1 --binByBinStat"

os.system(cmd)