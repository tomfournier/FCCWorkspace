import os
import importlib
# userConfig = importlib.import_module('userConfig')

# outputdir = userConfig.loc.COMBINE

cmd = f"singularity exec /eos/project/f/fccsw-web/www/analysis/auxiliary/combine-standalone_v9.2.1.sif bash -c 'pwd;ls -l;cd analysis/ZH/output/combine/mumu; text2workspace.py datacard.txt -o ws.root; combine -M MultiDimFit -v 10 --rMin 0.9 --rMax 1.1 --setParameters r=1 ws.root'"

os.system(cmd)