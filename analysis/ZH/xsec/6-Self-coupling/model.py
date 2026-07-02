import math

from package.func.self_coupling import PhysicsModel



#############################
### QUANTITIES DEFINITION ###
#############################

mH  = 125.1
Gmu = 1.16638e-5
vev = 1 / (2**0.5 * Gmu)

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
Ds_weak = {240: -7.1,  365: -3.7}
Ds_QED  = {240: -37.7, 365: +7.9}


def get_contribution(wilson_coeff, ecm, lbda=1):
    '''Get the contribution for the Wilson coefficient considered'''

    weak = table[wilson_coeff][ecm]['weak']
    qed  = table[wilson_coeff][ecm]['QED']
    bar  = table[wilson_coeff][ecm]['bar']

    ratio = sigma_weak[ecm] / sigma_LO[ecm] / lbda**2
    coeff = weak + qed + bar * math.log(240**2 / ecm**2)

    return ratio * coeff


print(get_contribution('Cphi', 240))


#####################
### PHYSICS MODEL ###
#####################

class SMEFT_1D(PhysicsModel):

    def __init__(self, wilson_coeff: str):
        self.wilson = wilson_coeff

    def doParametersOfInterest(self):
        '''Create POI and other parameters, and define the POI set.'''
        self.modelBuilder.doVar(f'A[{get_contribution(self.wilson, 240)}]')
        self.modelBuilder.doVar(f'B[{get_contribution(self.wilson, 365)}]')
        self.modelBuilder.out.var('A').setConstant(True)
        self.modelBuilder.out.var('B').setConstant(True)

        self.modelBuilder.doVar(f'{self.wilson}[0,-1,1]')
        self.modelBuilder.factory_(f'expr::{self.wilson}_scaling_240(\"1+@0*@1\", A, {self.wilson})')
        self.modelBuilder.factory_(f'expr::{self.wilson}_scaling_365(\"1+@0*@1\", B, {self.wilson})')
        self.modelBuilder.doSet('POI', ','.join([self.wilson]))

    def getYieldScale(self, process, energy):
        if process == 'sig': return f'{self.wilson}_scaling_{energy}'
        else: return 1

class SMEFT_2D(PhysicsModel):

    def __init__(self, wilson_coeff1: str, wilson_coeff2: str):
        self.wilson1 = wilson_coeff1
        self.wilson2 = wilson_coeff2


    def doParametersOfInterest(self):
        '''Create POI and other parameters, and define the POI set.'''

        for i, wilson in enumerate([self.wilson1, self.wilson2]):
            self.modelBuilder.doVar(f'A{i}[{get_contribution(wilson, 240)}]')
            self.modelBuilder.doVar(f'B{i}[{get_contribution(wilson, 365)}]')
            self.modelBuilder.out.var(f'A{i}').setConstant(True)
            self.modelBuilder.out.var(f'B{i}').setConstant(True)

            self.modelBuilder.doVar(f'{wilson}[0,-1,1]')
        self.modelBuilder.factory_(f'expr::2D_scaling_240(\"1 + @0*@1 + @2*@3\", A0, {self.wilson1}, A1, {self.wilson2})')
        self.modelBuilder.factory_(f'expr::2D_scaling_365(\"1 + @0*@1 + @2*@3\", B0, {self.wilson1}, B2, {self.wilson2})')
        self.modelBuilder.doSet('POI', ','.join([self.wilson1, self.wilson2]))

    def getYieldScale(self, process, energy):
        if process == 'sig': return f'2D_scaling_{energy}'
        else: return 1


smeft_Cphi  = SMEFT_1D('Cphi')
smeft_CphiD = SMEFT_1D('CphiD')
smeft_Cbox  = SMEFT_1D('Cbox')
