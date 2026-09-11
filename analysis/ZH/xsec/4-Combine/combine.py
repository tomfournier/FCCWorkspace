#################################
### STANDARD LIBRARIES IMPORT ###
#################################

import time

t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,
    ecm_multi=True,
    include_sels=True,
    combine=True,
    description='Datacard making script'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load user configuration and event utilities
from package.userConfig import loc
from package.config import H_DECAYS_FIT, get_process_dict, timer
from package.func.combine import do_combine



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecms, sels = arg.cat.split('-'), arg.ecm.split('-'), arg.sels.split('-')

mc_stats = arg.mc_stats    # Include MC statistical uncertainties (default: False, assume Poisson)
rebin    = arg.rebin       # Histogram rebinning factor (1 = no rebinning)
intLumi  = arg.intLumi     # Luminosity scaling factor for normalization
scales = {}                # Re-scale histograms (value per process)
only_asimov = False        # Apply scales only to asimov histograms

sig_procs_dict = {
    '240': get_process_dict(['ZH'], 240, h_decays=H_DECAYS_FIT),
    '365': get_process_dict(['ZH'], 365, h_decays=H_DECAYS_FIT),
}
bkg_procs_dict = {
    '240':{'*':  get_process_dict(['ZZ', 'WW', 'Zgamma', 'Rare'],       240)},
    '365':{'*':  get_process_dict(['ZZ', 'WW', 'Zgamma', 'Rare'],       365),
           'qq': get_process_dict(['ZZ', 'WW', 'Zgamma', 'Rare', 'tt'], 365)}
}

hist_names_dict = {
    'lep': ['zll_recoil_m_fit_high', 'zll_recoil_m_fit_low'],
    'had': ['zqq_m_recoil_m_tot_mva_fit_high_1D', 'zqq_m_recoil_m_tot_mva_fit_low_1D']
    # 'had': ['zqq_recoil_m_tot_fit_high', 'zqq_recoil_m_tot_fit_low']
}
# Category identifier
cats_template: list[str] = ['z_cat_high', 'z_cat_low']



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    for ecm in ecms:
        sig_procs = sig_procs_dict.get(ecm, {})
        if arg.rescale:
            lumi = (intLumi/10.8) if ecm=='240' else ((intLumi/3.12) if ecm=='365' else intLumi)
        else:
            lumi = 1
        if not sig_procs:
            LOGGER.warning('sig_procs is an empty dictionary')
        for cat in cats:
            bkg_procs_ecm = bkg_procs_dict.get(ecm, {})
            bkg_procs     = bkg_procs_ecm.get(cat, bkg_procs_ecm['*'])
            if not bkg_procs:
                LOGGER.warning('bkg_procs is an empty dictionary')
            categories = [c.replace('cat', cat) for c in cats_template]
            hist_names = hist_names_dict['had' if cat=='qq' else 'lep']

            # Define systematic uncertainties
            systs = {
                f'{proc}_norm':{
                    'type':'lnN',     # Log-normal uncertainty
                    'value':1.01,     # 1% normalization uncertainty
                    'procs':[proc]}   # Apply to this process
                for proc in bkg_procs.keys()
            }
            systs_procs = {}
            for sel in sels:
                inputDir  = loc.get('HIST_PROCESSED',   cat, ecm, sel)
                outputDir = loc.get('NOMINAL_DATACARD', cat, ecm, sel)

                do_combine(
                    inputDir, outputDir, hist_names,
                    categories, sig_procs, bkg_procs, systs,
                    systs_procs, rebin, lumi, scales,
                    only_asimov, mc_stats
                )
    return None


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
