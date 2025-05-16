import ROOT

print('Getting file')
f = ROOT.TFile('analysis/ZH/output/combine/mumu/higgsCombineTest.MultiDimFit.mH120.root')
print('Getting Fit results')
tree = f.Get('limit')
print('Getting limit')
r = tree.limit
print('Getting error')
err = tree.limitErr
print('Getting mh')
mh = tree.mh

print(f'r = {r} +/- {err}')
print(f'mh = {mh}')

f1 = ROOT.TFile('analysis/ZH/output/combine/mumu/multidimfitTest.root')
fit = f1.Get('fit_mdf')
process = f1.Get('ProcessID0')

print(f'fit = {fit}')
print(f'process = {process}')