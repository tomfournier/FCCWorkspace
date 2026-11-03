from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModel

from logger import get_logger
from path import PathObj

LOGGER = get_logger(__name__)



#############################
### QUANTITIES DEFINITION ###
#############################

# Higgs boson mass
mH  = 125.1
# Fermi's constant from muon decay
Gmu = 1.16638e-5
# Vacuum expectation value
vev = (1 / (2**0.5 * Gmu))**0.5

# SMEFT's contributions to the ZH production cross-section
table: dict[str, dict[int, dict[str, float]]] = {
    'Cphi':  {240: {'weak': -7.21e-3, 'QED': +0,       'bar': +0},
              365: {'weak': -9.98e-4, 'QED': +0,       'bar': +0}},
    'CphiD': {240: {'weak': +1.20e-1, 'QED': -1.95e-2, 'bar': -5.41e-3},
              365: {'weak': +1.12e-1, 'QED': +8.56e-3, 'bar': -5.75e-3}},
    'Cbox':  {240: {'weak': +8.80e-3, 'QED': -2.46e-3, 'bar': -1.12e-3},
              365: {'weak': +7.30e-3, 'QED': +2.36e-3, 'bar': -1.17e-3}}
}
xsec = {'LO': {240: 239.2, 365: 116.9}, 'NLO': {240: 194.4, 365: 121.1}, 'weak': {240: 232.1, 365: 113.2}}

# Wilson's coefficients label for plot
coeffs_label = {'r': r'$\mu_{ZH}$', 'Cphi': r'$C_{\phi}$', 'CphiD': r'$C_{\phi D}$', 'CBox':  r'$C_{\Box}$',}



########################
### HELPER FUNCTIONS ###
########################

def get_params(model: str):
    """Return the Wilson coefficients attached to a named exported model."""

    try:
        model_obj = globals()[model]
    except KeyError as exc:
        raise RuntimeError(f'Unknown model {model!r}') from exc

    wilson = getattr(model_obj, 'wilson', None)
    if wilson is None:
        raise RuntimeError(f'Model {model!r} does not define a wilson list')

    return wilson


def from_SMEFT(
        Cphi: float | int,
        CphiD: float | int,
        Cbox: float | int,
        lbda: float | int = 1e3
) -> float | int:
    '''Convert the values of Cphi, CphiD and Cbox to kappa_lambda'''
    Ckin = (CphiD/4 - Cbox)
    kappa = 1 + (vev/lbda)**2 * (3 * Ckin - 2 * (vev/mH)**2 * Cphi)
    return kappa


def get_kappa_err(
        Cphi_err: float | int,
        CphiD_err: float | int,
        Cbox_err: float | int,
        lbda: float | int = 1e3
) -> float | int:
    '''Convert the errors of Cphi, CphiD and Cbox to kappa_lambda error'''
    dCphi  = from_SMEFT(1, 0, 0, lbda) - 1
    dCphiD = from_SMEFT(0, 1, 0, lbda) - 1
    dCbox  = from_SMEFT(0, 0, 1, lbda) - 1

    LOGGER.debug(f'Using {dCphi = :.4f}, {dCphiD = :.4f}, {dCbox = :.4f}')

    dkappa = ((dCphi*Cphi_err)**2 + (dCphiD*CphiD_err)**2 + (dCbox*Cbox_err)**2)**0.5
    return dkappa


def to_kappa(
        res: dict[str, dict[str, float | int]],
        outputdir: PathObj | str | None = None,
        suffix: str = '',
        lbda: float | int = 1e3,
) -> tuple[float | int, float | int]:
    '''Convert the results from Cphi, CphiD and Cbox to kappa_lambda results'''

    import json

    cphi, cphid, cbox = res.get('Cphi', {}), res.get('CphiD', {}),  res.get('Cbox', {})

    Cphi,  Cphi_err  = cphi.get('best_fit',  0), cphi.get('err',  0)
    CphiD, CphiD_err = cphid.get('best_fit', 0), cphid.get('err', 0)
    Cbox,  Cbox_err  = cbox.get('best_fit',  0), cbox.get('err',  0)

    kappa_val = from_SMEFT(Cphi, CphiD, Cbox, lbda)
    kappa_err = get_kappa_err(Cphi_err, CphiD_err, Cbox_err, lbda)

    LOGGER.info(f'Converting to kappa framework:\n\tk_l = {kappa_val:.2f} +/- {kappa_err:.2%}')

    if outputdir is not None:
        if not isinstance(outputdir, PathObj): outputdir = PathObj(outputdir)
        out = outputdir / f'result_kappa{suffix}.json'
        out.write_text(json.dumps({'kappa': kappa_val, 'err': kappa_err}))
        LOGGER.info(f'Saving results at {out}')
    return kappa_val, kappa_err



#############################
### PHYSICS MODEL HELPERS ###
#############################

def get_coeff(wilson: str, ecm: int, lbda: float | int = 1, LO: bool = True) -> float:
    '''Get the contribution for the Wilson coefficient considered'''
    import math

    weak = table[wilson][ecm]['weak']
    qed  = table[wilson][ecm]['QED']
    bar  = table[wilson][ecm]['bar']

    if LO: ratio = xsec['weak'][ecm] / xsec['LO'][ecm]  / lbda**2
    else:  ratio = xsec['weak'][ecm] / xsec['NLO'][ecm] / lbda**2
    coeff = weak + qed + bar * math.log(240**2 / ecm**2)

    return ratio * coeff


def get_expr(prefix: str, wilsons: list[str], energy: int) -> str:
    """Build the RooFit expression for an arbitrary number of Wilson coefficients."""

    coeff_prefix = {240: "A", 365: "B"}
    if energy not in coeff_prefix:
        raise RuntimeError(f"Unsupported energy {energy}; expected 240 or 365")

    terms, expr = [f'@{2*i}*@{2*i + 1}' for i in range(len(wilsons))], '1'
    if terms: expr += '+' + '+'.join(terms)

    args = []
    for i, wilson in enumerate(wilsons):
        args.extend([f'{coeff_prefix[energy]}{i}', wilson])

    return f'expr::{prefix}_scaling_{energy}("{expr}", {", ".join(args)})'


def get_energy(bin_name: str) -> int:
    """Extract the energy from a bin name like low_ch1 or high_ch2."""

    prefix = bin_name.split("_", 1)[0]
    if prefix in ['high', 'low']:
        return 240 if prefix=='low' else (365 if prefix=='high' else -1)

    raise ValueError('Bin names should be <energy>_ch1 with energy in [high, low]')



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
            self.modelBuilder.doVar(f'A{i}[{get_coeff(wilson, 240, 1, True)}]')
            self.modelBuilder.doVar(f'B{i}[{get_coeff(wilson, 365, 1, True)}]')

            # Set the contributions as constant
            self.modelBuilder.out.var(f'A{i}').setConstant(True)
            self.modelBuilder.out.var(f'B{i}').setConstant(True)

            # Define the wilson coefficient
            self.modelBuilder.doVar(f'{wilson}[0,-2,2]')

        self.modelBuilder.factory_(get_expr('SMEFT', self.wilson, 240))
        self.modelBuilder.factory_(get_expr('SMEFT', self.wilson, 365))
        self.modelBuilder.doSet('POI', ','.join(self.wilson))


    def getYieldScale(self, bin_name, process):
        energy = get_energy(bin_name)
        if self.DC.isSignal[process]:
            return f'SMEFT_scaling_{energy}'
        return 1



########################
### MODEL DEFINITION ###
########################

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

class _SMEFT_CphiD_Cbox(SMEFT_NLO):
    wilson = ['CphiD', 'Cbox']

class _SMEFT_all(SMEFT_NLO):
    wilson = ['Cphi', 'CphiD', 'Cbox']


# Has to define the models like this
# Will get error if not
SMEFT_Cphi  = _SMEFT_Cphi()
SMEFT_CphiD = _SMEFT_CphiD()
SMEFT_Cbox  = _SMEFT_Cbox()

SMEFT_Cphi_CphiD = _SMEFT_Cphi_CphiD()
SMEFT_Cphi_Cbox  = _SMEFT_Cphi_Cbox()
SMEFT_CphiD_Cbox = _SMEFT_CphiD_Cbox()

SMEFT_all = _SMEFT_all()