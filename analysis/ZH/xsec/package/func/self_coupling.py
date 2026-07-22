import math

from abc import ABCMeta, abstractmethod

from HiggsAnalysis.CombinedLimit.PhysicsModel import (  # type:ignore
    PhysicsModel as CombinedPhysicsModel,
    PhysicsModelBase as CombinedPhysicsModelBase,
)

from ..logger import get_logger

LOGGER = get_logger(__name__)



#############################
### QUANTITIES DEFINITION ###
#############################

mH  = 125.1
Gmu = 1.16638e-5
vev = (1 / (2**0.5 * Gmu))**0.5

table: dict[str, dict[int, dict[str, float]]] = {
    'Cphi': {
        240: {'weak': -7.21e-3, 'QED': 0, 'bar': 0},
        365: {'weak': -9.98e-4, 'QED': 0, 'bar': 0}
    },
    'CphiD': {
        240: {'weak': 1.20e-1, 'QED': -1.95e-2, 'bar': -5.41e-3},
        365: {'weak': 1.12e-1, 'QED': +8.56e-3, 'bar': -5.75e-3}
    },
    'Cbox': {
        240: {'weak': 8.80e-3, 'QED': -2.46e-3, 'bar': -1.12e-3},
        365: {'weak': 7.30e-3, 'QED': +2.36e-3, 'bar': -1.17e-3}
    }
}

sigma_LO   = {240: 239.2, 365: 116.9}
sigma_NLO  = {240: 194.4, 365: 121.1}
sigma_weak = {240: 232.1, 365: 113.2}



########################
### HELPER FUNCTIONS ###
########################

def get_contribution(
        wilson_coeff: dict[str,
                           dict[int,
                                dict[str, float]]],
        ecm: int,
        lbda: float | int = 1,
        LO: bool = True) -> float:
    '''Get the contribution for the Wilson coefficient considered'''

    weak = table[wilson_coeff][ecm]['weak']
    qed  = table[wilson_coeff][ecm]['QED']
    bar  = table[wilson_coeff][ecm]['bar']

    if LO: ratio = sigma_weak[ecm] / sigma_LO[ecm]  / lbda**2
    else:  ratio = sigma_weak[ecm] / sigma_NLO[ecm] / lbda**2
    coeff = weak + qed + bar * math.log(240**2 / ecm**2)

    return ratio * coeff


def build_expression(prefix: str, wilsons: list[str], energy: int) -> str:
    """Build the RooFit expression for an arbitrary number of Wilson coefficients."""

    coeff_prefix = {240: "A", 365: "B"}
    if energy not in coeff_prefix:
        raise RuntimeError(f"Unsupported energy {energy}; expected 240 or 365")

    terms = [f'@{2 * i}*@{2 * i + 1}' for i in range(len(wilsons))]
    expr = '1'
    if terms:
        expr += ' + ' + ' + '.join(terms)

    args: list[str] = []
    for i, wilson in enumerate(wilsons):
        args.extend([f'{coeff_prefix[energy]}{i}', wilson])

    return f'expr::{prefix}_scaling_{energy}("{expr}", {", ".join(args)})'


def get_bin_energy(bin_name: str) -> int:
    """Extract the energy from a bin name like 240_ch1 or 365_ch2."""

    prefix = bin_name.split("_", 1)[0]
    if prefix == 'low':
        return 240
    if prefix == 'high':
        return 365

    try:
        energy = int(prefix)
    except ValueError as exc:
        raise RuntimeError(f"Cannot infer the energy from bin {bin_name!r}") from exc

    if energy not in (240, 365):
        raise RuntimeError(f"Unsupported energy {energy} in bin {bin_name!r}; expected 240 or 365")
    return energy


def kappa_from_SMEFT(
        Cphi: float | int,
        CphiD: float | int,
        Cbox: float | int,
        lbda: float | int = 1e3
         ) -> float | int:
    Ckin = (CphiD/4 - Cbox)
    kappa = 1 + (vev/lbda)**2 * (3 * Ckin - 2 * (vev/mH)**2 * Cphi)
    return kappa


def kappa_precision(
        Cphi: float | int,
        CphiD: float | int,
        Cbox: float | int,
        lbda: float | int = 1e3
         ) -> float | int:
    dCphi  = kappa_from_SMEFT(1, 0, 0, lbda) - 1
    dCphiD = kappa_from_SMEFT(0, 1, 0, lbda) - 1
    dCbox  = kappa_from_SMEFT(0, 0, 1, lbda) - 1

    LOGGER.debug(f'Using {dCphi = :.4f}, {dCphiD = :.4f}, {dCbox = :.4f}')

    dkappa = ((dCphi*Cphi)**2 + (dCphiD*CphiD)**2 + (dCbox*Cbox)**2)**0.5
    return dkappa


def get_parameters(model: str):
    """Return the Wilson coefficients attached to a named exported model."""

    try:
        model_obj = globals()[model]
    except KeyError as exc:
        raise RuntimeError(f'Unknown model {model!r}') from exc

    wilson = getattr(model_obj, 'wilson', None)
    if wilson is None:
        raise RuntimeError(f'Model {model!r} does not define a wilson list')

    return wilson




###############################
### HELPER CLASS DEFINITION ###
###############################

### Class that takes care of building a physics model by combining individual channels and processes together
### Things that it can do:
###   - define the parameters of interest (in the default implementation , "r")
###   - define other constant model parameters (e.g., "MH")
###   - yields a scaling factor for each pair of bin and process (by default, constant for background and linear in "r" for signal)
###   - possibly modifies the systematical uncertainties (does nothing by default)

class PhysicsModelBase(CombinedPhysicsModelBase, metaclass=ABCMeta):
    def setModelBuilder(self, modelBuilder):
        "Connect to the ModelBuilder to get workspace, datacard and options. Should not be overloaded."
        self.modelBuilder = modelBuilder
        self.DC = modelBuilder.DC
        self.options = modelBuilder.options

    def setPhysicsOptions(self, physOptions):
        "Receive a list of strings with the physics options from command line"
        return None

    @abstractmethod
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""

    def preProcessNuisances(self, nuisances):
        "receive the usual list of (name,nofloat,pdf,args,errline) to be edited"
        return None

    def getYieldScale(self, bin_name, process):
        "Return the name of a RooAbsReal to scale this yield by or the two special values 1 and 0 (don't scale, and set to zero)"
        return "r" if self.DC.isSignal[process] else 1

    def getChannelMask(self, bin_name):
        "Return the name of a RooAbsReal to mask the given bin (args != 0 => masked)"
        name = "mask_%s" % bin_name
        # Check that the mask expression doesn't exist already, it might do
        # if it was already defined in the datacard
        if not self.modelBuilder.out.arg(name):
            self.modelBuilder.doVar("%s[0]" % name)
        return name

    def done(self):
        "Called after creating the model, except for the ModelConfigs"
        return None



############################
### CLASS BASE DEFINTION ###
############################

class PhysicsModel(CombinedPhysicsModel):
    """Example class with signal strength as only POI"""

    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        self.modelBuilder.doVar("r[1,0,20]")
        self.modelBuilder.doSet("POI", "r")
        # --- Higgs Mass as other parameter ----
        if self.options.mass != 0:
            if self.modelBuilder.out.var("MH"):
                var = self.modelBuilder.out.var("MH")
                var.removeMin()
                var.removeMax()
                var.setVal(self.options.mass)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass)



#####################
### PHYSICS MODEL ###
#####################

class SMEFT_NLO(PhysicsModel):

    wilson: list[str] = []

    def __init__(self, wilson_coeff: str | list[str] | None = None):
        super().__init__()
        if wilson_coeff is None:
            wilson_coeff = self.wilson
        if isinstance(wilson_coeff, str):
            self.wilson = [wilson_coeff]
        elif isinstance(wilson_coeff, list):
            self.wilson = wilson_coeff
        else:
            raise TypeError('Only accept str or list[str] as input')


    def doParametersOfInterest(self):
        '''Create POI and other parameters, and define the POI set.'''

        for i, wilson in enumerate(self.wilson):
            # Define the contribution for the corresponding wilson coefficient
            self.modelBuilder.doVar(f'A{i}[{get_contribution(wilson, 240, 1, True)}]')
            self.modelBuilder.doVar(f'B{i}[{get_contribution(wilson, 365, 1, True)}]')

            # Set the contributions as constant
            self.modelBuilder.out.var(f'A{i}').setConstant(True)
            self.modelBuilder.out.var(f'B{i}').setConstant(True)

            # Define the wilson coefficient
            self.modelBuilder.doVar(f'{wilson}[0,-2,2]')

        self.modelBuilder.factory_(build_expression('SMEFT', self.wilson, 240))
        self.modelBuilder.factory_(build_expression('SMEFT', self.wilson, 365))
        self.modelBuilder.doSet('POI', ','.join(self.wilson))


    def getYieldScale(self, bin_name, process):
        energy = get_bin_energy(bin_name)
        if self.DC.isSignal[process]:
            return f'SMEFT_scaling_{energy}'
        return 1


class _SMEFT_Cphi(SMEFT_NLO):
    wilson = ['Cphi']


class _SMEFT_CphiD(SMEFT_NLO):
    wilson = ['CphiD']


class _SMEFT_Cbox(SMEFT_NLO):
    wilson = ['Cbox']


class _SMEFT_Cphi_CphiD(SMEFT_NLO):
    wilson = ['Cphi', 'CphiD']


class _SMEFT_Cphi_Cbox(SMEFT_NLO):
    wilson = ['Cphi', 'Cbox']

class _SMEFT_Cbox_Cphi(SMEFT_NLO):
    wilson = ['Cbox', 'Cphi']

class _SMEFT_CphiD_Cbox(SMEFT_NLO):
    wilson = ['CphiD', 'Cbox']


class _SMEFT_all(SMEFT_NLO):
    wilson = ['Cphi', 'Cbox', 'CphiD']


SMEFT_Cphi  = _SMEFT_Cphi()
SMEFT_CphiD = _SMEFT_CphiD()
SMEFT_Cbox  = _SMEFT_Cbox()
SMEFT_Cphi_CphiD = _SMEFT_Cphi_CphiD()
SMEFT_Cphi_Cbox  = _SMEFT_Cphi_Cbox()
SMEFT_Cbox_Cphi  = _SMEFT_Cbox_Cphi()
SMEFT_CphiD_Cbox = _SMEFT_CphiD_Cbox()
SMEFT_all = _SMEFT_all()
