import os
import ROOT
import importlib, time

t1 = time.time()

userConfig = importlib.import_module('userConfig')
from userConfig import samples, loc, final_state

inputdir  = loc.HIST_PREPROCESSED
outputdir = loc.HIST_PROCESSED
procs = samples

#__________________________________________________________
def get_hist(hName, proc):
    print(f'----->[Info] Getting histogram from \n\t {inputdir}/{proc}.root')
    f = ROOT.TFile(f"{inputdir}/{proc}.root")
    h = f.Get(hName)
    h.SetDirectory(0)
    f.Close()
    return h

#__________________________________________________________
def unroll(hist, hName):
    # Get the number of bins in X and Y
    n_bins_x = hist.GetNbinsX()
    n_bins_y = hist.GetNbinsY()

    # Create a 1D histogram to hold the unrolled data
    n_bins_1d = n_bins_x * n_bins_y
    h1 = ROOT.TH1D("h1", "1D Unrolled Histogram", n_bins_1d, 0.5, n_bins_1d + 0.5)

    # Loop over all bins in the 2D histogram
    for bin_x in range(1, n_bins_x + 1):  # Bin indexing starts at 1
        for bin_y in range(1, n_bins_y + 1):
            # Calculate the global bin number for the 1D histogram
            bin_1d = (bin_y - 1) * n_bins_x + bin_x

            # Get the content and error from the 2D histogram
            content = hist.GetBinContent(bin_x, bin_y)
            error = hist.GetBinError(bin_x, bin_y)

            # Set the content and error in the 1D histogram
            h1.SetBinContent(bin_1d, content)
            h1.SetBinError(bin_1d, error)
    h1.SetName(hName)
            
    return h1


if not os.path.isdir(f'{outputdir}'):
    os.system(f'mkdir -p {outputdir}')

print('----->[Info] Processing histograms for Combine')
for proc in procs:
    print(f'----->[Info] Processing {proc}')
    h = get_hist(f'{final_state}_recoil_m_mva', proc)
    
    print('----->[Info] Unrolling histogram')
    h = unroll(h, f'{final_state}_recoil_m_mva')
    print(f'----->[Info] {proc} histogram processed')

    print(f'----->[Info] Saving histogram')
    f = ROOT.TFile(f"{outputdir}/{proc}.root", "RECREATE")
    h.Write()
    f.Close()
    print(f'----->[Info] Saved histogram in {outputdir}/{proc}.root')

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
