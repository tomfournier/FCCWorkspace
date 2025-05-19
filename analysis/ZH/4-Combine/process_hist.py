import os
import ROOT
import importlib

userConfig = importlib.import_module('userConfig')

final_state = userConfig.final_state
ecm = userConfig.ecm

inputdir = userConfig.loc.COMBINE_HIST
outputdir = userConfig.loc.COMBINE_PROC

ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if final_state=='ee' else f"wzp6_ee_mumu_ecm{ecm}"

procs = {
    #signal
    f"wzp6_ee_{final_state}H_ecm{ecm}",
    #background: 
    f"p8_ee_WW_ecm{ecm}", f"p8_ee_ZZ_ecm{ecm}",
    ee_ll,
    f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}", f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
    f"wzp6_gaga_{final_state}_60_ecm{ecm}",
    f"wzp6_ee_tautau_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}", 
    f"wzp6_ee_nuenueZ_ecm{ecm}"
}

def get_hist(hName, proc):
    f = ROOT.TFile(f"{inputdir}/{proc}.root")
    h = f.Get(hName)
    h.SetDirectory(0)
    f.Close()
    return h

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
    os.system(f'mkdir {outputdir}')

print('----->[Info] Processing histograms for Combine')
for proc in procs:
    print(f'----->[Info] Processing {proc}')
    print('----->[Info] Getting histogram')
    h = get_hist(f'{final_state}_recoil_m_mva', proc)
    
    print('----->[Info] Unrolling histogram')
    h = unroll(h, f'{final_state}_recoil_m_mva')
    print(f'----->[Info] {proc} histogram processed')

    print(f'----->[Info] Saving histogram')
    f = ROOT.TFile(f"{outputdir}/{proc}.root", "RECREATE")
    h.Write()
    f.Close()
    print(f'----->[Info] Saved histogram in {outputdir}/{proc}.root')