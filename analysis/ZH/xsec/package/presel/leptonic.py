"""Leptonic channel preselection and training functions.

Provides complete workflow for building Z->ll candidates in ee and mumu channels:
- Collection aliasing and baseline lepton selection
- Z boson reconstruction with recoil mass computation
- Angular correlations and kinematic observables
- Cutflow tracking for event selection efficiency

Functions:
- `_setup_alias()`: Define collection aliases for leptons and MC truth.
- `_lesp_properties()`: Baseline lepton kinematics and isolation cuts.
- `_recoil_builder()`: Z candidate construction with H->ll veto.
- `_Z_kinematics()`: Z boson four-vector and recoil quantities.
- `_lead_sub_properties()`: Leading/subleading lepton ordering.
- `_additional_variables()`: Angular, energy, and discriminant variables.
- `_cutflow()`: Event count bookkeeping.
- `training_ll()`: Build training dataframe (no cutflow).
- `presel_ll()`: Apply preselection with full cutflow chain.
"""

########################
### HELPER FUNCTIONS ###
########################

def _setup_alias(df, cat: str):
    """Attach collection aliases based on final state.

    Args:
        df: RDataFrame-like object used throughout the selection chain.
        cat (str): Final-state category, either 'mumu' or 'ee'.

    Returns:
        The input dataframe with lepton, MC association, and particle aliases defined.

    Raises:
        ValueError: If an unsupported category is provided.
    """
    # Alias for lepton collections based on final state
    if cat == 'mumu':
        df = df.Alias('Lepton0', 'Muon#0.index')
    elif cat == 'ee':
        df = df.Alias('Lepton0', 'Electron#0.index')
    else:
        raise ValueError(f'cat {cat} not supported')
    
    # Alias for MC truth matching and particle collections
    df = df.Alias('MCRecoAssociations0', 'MCRecoAssociations#0.index')
    df = df.Alias('MCRecoAssociations1', 'MCRecoAssociations#1.index')
    df = df.Alias('Particle0', 'Particle#0.index')
    df = df.Alias('Particle1', 'Particle#1.index')
    df = df.Alias('Photon0', 'Photon#0.index')
    return df


def _lesp_properties(df):
    """Define baseline lepton collections and isolation selections.

    Computes lepton kinematics (p, theta, phi) for all leptons before and after
    momentum cut (p > 20 GeV). Applies cone isolation (0.01 < dR < 0.5) and
    selects isolated leptons (Irel < 0.25) for downstream analysis.

    Args:
        df: RDataFrame-like object with reconstructed particles.

    Returns:
        The dataframe with baseline lepton collections and kinematic properties:
        - All leptons: leps_all, leps_all_p, leps_all_theta, leps_all_phi, leps_all_iso
        - Selected leptons: leps, leps_p, leps_theta, leps_phi, leps_iso
        - Isolated leptons: leps_sel_iso with count leps_sel_no
    """
    # Define all lepton properties (before cuts)
    df = df.Define('leps_all',       'FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)')
    df = df.Define('leps_all_p',     'FCCAnalyses::ReconstructedParticle::get_p(leps_all)')
    df = df.Define('leps_all_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(leps_all)')
    df = df.Define('leps_all_phi',   'FCCAnalyses::ReconstructedParticle::get_phi(leps_all)')
    df = df.Define('leps_all_q',     'FCCAnalyses::ReconstructedParticle::get_charge(leps_all)')
    df = df.Define('leps_all_no',    'FCCAnalyses::ReconstructedParticle::get_n(leps_all)')
    df = df.Define('leps_all_iso',   'FCCAnalyses::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)') 
    
    # Apply momentum cut (p > 20 GeV) to reduce soft backgrounds
    df = df.Define('leps',         'FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)')
    df = df.Define('leps_p',       'FCCAnalyses::ReconstructedParticle::get_p(leps)')
    df = df.Define('leps_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(leps)')
    df = df.Define('leps_phi',     'FCCAnalyses::ReconstructedParticle::get_phi(leps)')
    df = df.Define('leps_q',       'FCCAnalyses::ReconstructedParticle::get_charge(leps)')
    df = df.Define('leps_no',      'FCCAnalyses::ReconstructedParticle::get_n(leps)')
    df = df.Define('leps_iso',     'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    # Select isolated leptons: Irel < 0.25 (relative isolation ratio)
    df = df.Define('leps_sel_iso', 'FCCAnalyses::sel_isol(0.25)(leps, leps_iso)')
    df = df.Define('leps_sel_no',  'leps_sel_iso.size()') 
    return df


def _recoil_builder(df, ecm: int):
    """Build Z candidates and veto Higgs-like leptons.

    Two-step Z reconstruction: First veto H->ll (mass 125±3 GeV), then build best
    Z candidate from remaining leptons using mass~91.2 GeV and recoil~125 GeV.
    Removes Z leptons from full particle collection for subsequent calculations.

    Args:
        df: RDataFrame-like object containing lepton collections.
        ecm (int): Center-of-mass energy in GeV used for resonance builder.

    Returns:
        The dataframe with Z candidate (zll), constituent leptons (zll_leps),
        and particle collection excluding leptons (rps_no_leps).
    """
    # Veto H->mumu/ee candidate leptons (mass window: 125 ± 3 GeV)
    df = df.Define('zbuilder_result_Hll', f'FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)'
                    '(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll_Hll',             'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[0]}') # the Z
    df = df.Define('zll_Hll_m',           'FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]')
    df = df.Define('zll_leps_Hll',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[1],zbuilder_result_Hll[2]}') # the leptons
    df = df.Define('zll_leps_dummy',      'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}') # placeholder for empty vector
    # Remove H-candidate leptons if reconstructed mass falls in Higgs window
    df = df.Define('leps_to_remove',      'return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy')
    df = df.Define('leps_good',           'FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)') 

    # Build Z resonance from good leptons (mass ~ 91.2 GeV, recoil ~ 125 GeV)
    # resonanceBuilder returns: [0] di-lepton system, [1,2] individual leptons
    df = df.Define('zbuilder_result', f'FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0.4, {ecm}, false)'
                    '(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll',             'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}') # the Z
    df = df.Define('zll_leps',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}') # the leptons

    # Remove Z leptons from full particle collection for recoil-based observables
    df = df.Define('rps_no_leps', 'FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, zll_leps)')
    return df


def _Z_kinematics(df, ecm: int):
    """Compute Z-boson kinematics and recoil mass.

    Extracts Z four-vector properties (mass, momentum, angles) and computes
    recoil mass (Higgs candidate). Categorizes Z leptons by polar angle for
    geometric classification.

    Args:
        df: RDataFrame-like object containing a Z candidate (zll).
        ecm (int): Center-of-mass energy in GeV.

    Returns:
        The dataframe with Z properties (zll_m, zll_p, zll_pT, zll_theta, zll_phi),
        recoil mass (zll_recoil_m), and polar angle category (zll_category).
    """
    # Z boson kinematics
    df = df.Define('zll_m',     'FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]')
    df = df.Define('zll_p',     'FCCAnalyses::ReconstructedParticle::get_p(zll)[0]')
    df = df.Define('zll_pT',    'FCCAnalyses::ReconstructedParticle::get_pt(zll)[0]')
    df = df.Define('zll_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]')
    df = df.Define('zll_phi',   'FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]')
    
    # Recoil mass: invariant mass of system recoiling against Z (Higgs candidate)
    df = df.Define('zll_recoil',   f'FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)')
    df = df.Define('zll_recoil_m',  'FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]')
    # Polar angle categorization: 0.8 < |cos(theta)| < 2.34 (barrel vs endcap)
    df = df.Define('zll_category',  'FCCAnalyses::polarAngleCategorization(0.8, 2.34)(zll_leps)')
    return df


def _lead_sub_properties(df):
    """Extract leading/subleading lepton kinematics ordered by momentum.

    Identifies and indexes the higher/lower momentum leptons from Z decay,
    extracting their individual kinematic properties for MVA training and
    analysis-level discriminants.

    Args:
        df: RDataFrame-like object containing Z-lepton four-vectors (zll_leps).

    Returns:
        The dataframe with leading/subleading lepton indices and properties:
        leading_p, leading_pT, leading_theta, leading_phi and subleading equivalents.
    """
    # Individual Z lepton kinematics with leading/subleading ordering by momentum
    df = df.Define('zll_leps_p',       'FCCAnalyses::ReconstructedParticle::get_p(zll_leps)')
    df = df.Define('zll_leps_pT',      'FCCAnalyses::ReconstructedParticle::get_pt(zll_leps)')
    df = df.Define('zll_leps_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)')
    df = df.Define('zll_leps_phi',     'FCCAnalyses::ReconstructedParticle::get_phi(zll_leps)')
    # Identify which lepton has higher momentum
    df = df.Define('leading_p_idx',    '(zll_leps_p[0] > zll_leps_p[1]) ? 0 : 1')
    df = df.Define('subleading_p_idx', '(zll_leps_p[0] > zll_leps_p[1]) ? 1 : 0')

    # Extract leading/subleading properties
    df = df.Define('leading_p',        'zll_leps_p[leading_p_idx]')
    df = df.Define('leading_pT',       'zll_leps_pT[leading_p_idx]')
    df = df.Define('subleading_p',     'zll_leps_p[subleading_p_idx]')
    df = df.Define('subleading_pT',    'zll_leps_pT[subleading_p_idx]')

    df = df.Define('leading_theta',    'zll_leps_theta[leading_p_idx]')
    df = df.Define('subleading_theta', 'zll_leps_theta[subleading_p_idx]')
    df = df.Define('leading_phi',      'zll_leps_phi[leading_p_idx]')
    df = df.Define('subleading_phi',   'zll_leps_phi[subleading_p_idx]')
    return df


def _additional_variables(df, ecm: int):
    """Compute angular correlations, energy observables, and kinematic discriminants.

    Calculates three-body kinematics (acoplanarity, acolinearity, deltaR between
    leptons), Higgsstrahlungness discriminant from Z/recoil masses, and missing
    energy quantities for identifying Z+missing energy backgrounds.

    Args:
        df: RDataFrame-like object with Z candidate and remaining particles.
        ecm (int): Center-of-mass energy in GeV.

    Returns:
        The dataframe with angular variables (acoplanarity, acolinearity, deltaR),
        Higgsstrahlungness (H), visible/missing energy (visibleEnergy, missingMass),
        and missing energy angle (cosTheta_miss).
    """
    # Angular correlation variables
    df = df.Define('acoplanarity', 'FCCAnalyses::acoplanarity(zll_leps)')
    df = df.Define('acolinearity', 'FCCAnalyses::acolinearity(zll_leps)')
    df = df.Define('deltaR',       'FCCAnalyses::deltaR(zll_leps)')

    # Higgsstrahlungness: discriminant from Z mass and recoil mass ratio
    # Favors Z-strahlung topology (H recoiling against Z) over other processes
    df = df.Define('H', 'FCCAnalyses::Higgsstrahlungness(zll_m, zll_recoil_m)')

    # Visible energy (excluding Z candidate leptons)
    df = df.Define('visibleEnergy', 'FCCAnalyses::visibleEnergy(rps_no_leps)')
    # Missing energy and missing mass to identify invisible Higgs or other missing particle signatures
    df = df.Define('missingEnergy', f'FCCAnalyses::missingEnergy({ecm}, ReconstructedParticles)')
    df = df.Define('cosTheta_miss', 'FCCAnalyses::get_cosTheta_miss(missingEnergy)')
    df = df.Define('missingMass',   f'FCCAnalyses::missingMass({ecm}, ReconstructedParticles)')

    return df


def _cutflow(df, cut: str):
    """Record event count at each cutflow stage.

    Snapshot current event count and store as a constant column for efficiency tracking.
    Used to measure selection acceptance and background rejection.

    Args:
        df: RDataFrame-like object at a given selection step.
        cut (str): Column name used to store the event count (e.g., 'cut0', 'cut1').

    Returns:
        The dataframe with a constant column holding the event count value.
    """
    # Count events passing all previous cuts in the chain
    n  = df.Count().GetValue()
    # Store count as constant column in dataframe for later retrieval
    df = df.Define(cut, str(n))
    return df



######################
### MAIN FUNCTIONS ###
######################

#_______________________________________
def training_ll(df, cat: str, ecm: int):
    """Build training dataframe for leptonic channels without cutflow.

    Applies baseline cuts and derives all kinematic features for MVA training.
    Baseline cuts: ≥1 lepton with isolation, ≥2 opposite-sign leptons.
    Includes Z mass/momentum cuts (currently disabled for dataset uniformity).
    Recommended for BDT training and systematic studies.

    Args:
        df: Input RDataFrame-like object to augment.
        cat (str): Final-state category, either 'mumu' or 'ee'.
        ecm (int): Center-of-mass energy in GeV (240 or 365).

    Returns:
        The dataframe with complete kinematic feature set ready for training:
        lepton properties, Z kinematics, angular variables, energy observables,
        and Higgsstrahlungness discriminant.
    """

    df = _setup_alias(df, cat)
    df = _lesp_properties(df)

    ##########
    ### CUT 0: all events
    ##########

    ##########
    ### CUT 1: at least one lepton and at least one isolated lepton (I_rel < 0.25)
    ##########
    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0')

    ##########
    ### CUT 2: at least 2 leptons with opposite-sign
    ##########
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()')

    # Build Z candidate with H veto, extract all kinematic observables
    df = _recoil_builder(df, ecm)
    df = _Z_kinematics(df, ecm)
    df = _lead_sub_properties(df)
    df = _additional_variables(df, ecm)
    

    ##########
    ### CUT 3: Z mass window
    ##########
    # df = df.Filter('zll_m > 86 && zll_m < 96') 

    ##########
    ### CUT 4: Z momentum (CoM dependent)
    ##########
    # if ecm == 240:   df = df.Filter('zll_p > 20 && zll_p < 70')   # 240 GeV
    # elif ecm == 365: df = df.Filter('zll_p > 50 && zll_p < 150')  # 365 GeV
    
    return df

#_______________________________________________________
def presel_ll(df, cat: str, ecm: int, ww: bool = False):
    """Apply leptonic preselection with full cutflow tracking.

    Executes complete preselection chain with event counting at each stage:
    - cut0: All events
    - cut1: ≥1 lepton, ≥1 isolated lepton (Irel < 0.25)
    - cut2: ≥2 opposite-sign leptons
    - Derives kinematic features: Z mass/momentum, recoil, angles, energy

    Args:
        df: RDataFrame-like object to filter.
        cat (str): Final-state category, either 'mumu' or 'ee'.
        ecm (int): Center-of-mass energy in GeV (240 or 365).
        ww (bool, optional): If True, veto leptonic WW events (WW->ll backgrounds). Defaults to False.

    Returns:
        The dataframe with applied preselection filters and cutflow columns
        (cut0, cut1, cut2) plus all kinematic feature variables.
    """

    # Filter out leptonic WW events if WW flag is set
    if ww:
        df = df.Define('ww_leptonic', 'FCCAnalyses::is_ww_leptonic(Particle, Particle1)')
        df = df.Filter('!ww_leptonic')

    df = _setup_alias(df, cat)
    df = _lesp_properties(df)

    ##########
    ### Define cumulative gates for all events (no intermediate filtering)
    ##########
    df = _cutflow(df, 'cut0')

    ##########
    ### CUT 1: at least one lepton and at least one isolated lepton (I_rel < 0.25)
    ##########
    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0', 'cut1')
    df = _cutflow(df, 'cut1')

    ##########
    ### CUT 2: at least 2 leptons with opposite-sign
    ##########
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()', 'cut2')
    df = _cutflow(df, 'cut2')

    # Build Z candidate with H veto, extract all kinematic observables
    df = _recoil_builder(df, ecm)
    df = _Z_kinematics(df, ecm)
    df = _lead_sub_properties(df)
    df = _additional_variables(df, ecm)

    ##########
    ### CUT 3: Z mass window
    ##########
    # df = df.Filter('zll_m > 86 && zll_m < 96')
    # df = _cutflow(df, 'cut3')

    ##########
    ### CUT 4: Z momentum (CoM dependent)
    ##########
    # if ecm == 240:   df = df.Filter('zll_p > 20 && zll_p < 70')   # 240 GeV
    # elif ecm == 365: df = df.Filter('zll_p > 50 && zll_p < 150')  # 365 GeV
    # df = _cutflow(df, 'cut4')

    return df