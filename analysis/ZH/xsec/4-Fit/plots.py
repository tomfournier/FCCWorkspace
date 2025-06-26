import time, argparse
t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', choices=[10.8, 3.1], type=float, default=10.8)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
arg = parser.parse_args()

import importlib
userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, h_decays

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from tools.plotting import makePlot, PlotDecays

ecm, lumi = arg.ecm, arg.lumi

h_decays_nom, h_decays_oth = ['bb', 'cc', 'gg', 'ss', 'tautau', 'WW', 'aa'], ['bb', 'mumu', 'ZZ', 'Za', 'inv']
procs = [f"ZH", "WW", "ZZ", "Zgamma", "Rare"]

for cat in ['mumu', 'ee']:
        print(f'\n----->[Info] Making plots for {cat} channel\n')

        sel      = select(arg.recoil120, arg.miss, arg.bdt)
        inputDir = get_loc(loc.BIAS_DATACARD, cat, ecm, sel)
        outDir   = get_loc(loc.PLOTS_BIAS, cat, ecm, sel)

        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays, xMin=0, xMax=170, yMin=0.99, yMax=1.045, 
                   xLabel="High- and low-mass recoil", yLabel="Events", logY=False)
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays_nom, xMin=0, xMax=170, yMin=0.99, yMax=1.045, 
                   xLabel="High- and low-mass recoil", yLabel="Events", logY=False, outName='nominal')
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays_oth, xMin=0, xMax=170, yMin=0.99, yMax=1.045, 
                   xLabel="High- and low-mass recoil", yLabel="Events", logY=False, outName='other')
        
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays, xMin=145, xMax=167, yMin=0.97, yMax=1.04, 
                   xLabel="High-mass recoil", yLabel="Events", logY=False)
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays_nom, xMin=145, xMax=167, yMin=0.97, yMax=1.04, 
                   xLabel="High-mass recoil", yLabel="Events", logY=False, outName='nominal')
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays_oth, xMin=145, xMax=167, yMin=0.97, yMax=1.04, 
                   xLabel="High-mass recoil", yLabel="Events", logY=False, outName='other')
        
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays, xMin=0, xMax=105, yMin=0.997, yMax=1.005, 
                   xLabel="Low-mass recoil", yLabel="Events", logY=False)
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays_nom, xMin=0, xMax=105, yMin=0.997, yMax=1.005, 
                   xLabel="Low-mass recoil", yLabel="Events", logY=False, outName='nominal')
        PlotDecays(f'{cat}_data', inputDir, outDir, h_decays_oth, xMin=0, xMax=105, yMin=0.997, yMax=1.005, 
                   xLabel="Low-mass recoil", yLabel="Events", logY=False, outName='other')
        
        for tar in h_decays:
                makePlot(inputDir, outDir, cat, procs, target=tar, outName=f"bias_{tar}", xMin=0, xMax=200, yMin=0, 
                        ymin=0.95, ymax=1.05, xLabel="High- and low-mass recoil", yLabel="Events", logY=False)
                makePlot(inputDir, outDir, cat, procs, target=tar, outName=f"bias_{tar}", xMin=145, xMax=167, yMin=0, 
                        ymin=0.9, ymax=1.1, xLabel="High-mass recoil", yLabel="Events", logY=False)
                makePlot(inputDir, outDir, cat, procs, target=tar, outName=f"bias_{tar}", xMin=0, xMax=105, yMin=0, 
                        ymin=0.995, ymax=1.005, xLabel="Low-mass recoil", yLabel="Events", logY=False)
        
print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')