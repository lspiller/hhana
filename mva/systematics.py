# rootpy imports
from rootpy.tree import Cut

# local imports
from . import log; log = log[__name__]

class SYSTEMATICS_CATEGORIES:
    TAUS, \
    JETS, \
    WEIGHTS, \
    NORMALIZATION = range(4)

# WIP:
class Systematic(object):

    def __init__(self, name, variations=None):

        if variations is None:
            self.variations = ('UP', 'DOWN')
        else:
            if not isinstance(variations, (list, tuple)):
                variations = (variations,)
            self.variations = variations
        self.name = name

    def __iter__(self):

        for var in self.variations:
            yield '%s_%s' % (self.name, var)


SYSTEMATICS_COMMON = {
#TEMP    'TAU_TRIGGER': (('TAU_TRIGGER_SYST_UP',), ('TAU_TRIGGER_SYST_DOWN',)),
    'TAU_RECO': (('TAU_RECO_UP',), ('TAU_RECO_DOWN',)),
    'TAU_ELEOLR': (('TAU_ELEOLR_UP',), ('TAU_ELEOLR_DOWN',)),
    'TAU_ID': (('TAU_ID_UP',), ('TAU_ID_DOWN',)),
#    'QCD_FIT': (('QCDFIT_UP',), ('QCDFIT_DOWN',)),
#    'Z_FIT': (('ZFIT_UP',), ('ZFIT_DOWN',)),
#    'QCD_SHAPE': (('QCDSHAPE_UP',), ('QCDSHAPE_DOWN',)),
# TAU SYSTEMATICS
    'TAU_TRIGGER': (('TAU_TRIGGER_DOWN',), ('TAU_TRIGGER_UP',)),
#TEMP    'TAU_TRIGGER_STATDATA': (('TAU_TRIGGER_STATDATA_UP',), ('TAU_TRIGGER_STATDATA_DOWN',)),
#    'TAU_TRIGGER_SYST': (('TAU_TRIGGER_SYST_UP',), ('TAU_TRIGGER_SYST_DOWN',)),
#TEMP    'TAU_TRIGGER_STATMC': (('TAU_TRIGGER_STATMC_UP',), ('TAU_TRIGGER_STATMC_DOWN',)),
    'TAU_TES_DETECTOR': (('TAUS_TRUEHADTAU_SME_TES_DETECTOR_1_up',), ('TAUS_TRUEHADTAU_SME_TES_DETECTOR_1_down',)),
    'TAU_TES_MODEL': (('TAUS_TRUEHADTAU_SME_TES_MODEL_1_up',), ('TAUS_TRUEHADTAU_SME_TES_MODEL_1_down',)),
    'TAU_TES_INSITU': (('TAUS_TRUEHADTAU_SME_TES_INSITU_1_up',), ('TAUS_TRUEHADTAU_SME_TES_INSITU_1_down',)),
# MET SYSTEMATICS
    'MET_SoftTrk_ResoPara': (('MET_SoftTrk_ResoPara',),),
    'MET_SoftTrk_ResoPerp': (('MET_SoftTrk_ResoPerp',),),
    'MET_SoftTrk_Scale': (('MET_SoftTrk_ScaleUp',), ('MET_SoftTrk_ScaleDown',)),
# JET SYSTEMATICS
    'JET_BJES_Response': (('JET_BJES_Response_1_down',), ('JET_BJES_Response_1_up',)),
    'JET_Effective1': (('JET_EffectiveNP_1_1_down',), ('JET_EffectiveNP_1_1_up',)),
    'JET_Effective2': (('JET_EffectiveNP_2_1_down',), ('JET_EffectiveNP_2_1_up',)),
    'JET_Effective3': (('JET_EffectiveNP_3_1_down',), ('JET_EffectiveNP_3_1_up',)),
    'JET_Effective4': (('JET_EffectiveNP_4_1_down',), ('JET_EffectiveNP_4_1_up',)),
    'JET_Effective5': (('JET_EffectiveNP_5_1_down',), ('JET_EffectiveNP_5_1_up',)),
    'JET_Effective6': (('JET_EffectiveNP_6restTerm_1_down',), ('JET_EffectiveNP_6restTerm_1_up',)),
    'JET_Eta_Modelling': (('JET_EtaIntercalibration_Modelling_1_down',), ('JET_EtaIntercalibration_Modelling_1_up',)),
    'JET_Eta_NonClosure': (('JET_EtaIntercalibration_NonClosure_1_down',), ('JET_EtaIntercalibration_NonClosure_1_up',)),
    'JET_Eta_Stat': (('JET_EtaIntercalibration_TotalStat_1_down',), ('JET_EtaIntercalibration_TotalStat_1_up',)),
    'JET_Flavor_Comp': (('JET_Flavor_Composition_1_down',), ('JET_Flavor_Composition_1_up',)),
    'JET_Flavor_Resp': (('JET_Flavor_Response_1_up',), ('JET_Flavor_Response_1_down',)),
    'JET_JER_SINGLE_NP_1up': (('JET_JER_SINGLE_NP_1up',)),
    'JET_PU_Mu': (('JET_Pile_up_OffsetMu_1_down',), ('JET_Pile_up_OffsetMu_1_up',)),
    'JET_PU_NPV': (('JET_Pile_up_OffsetNPV_1_down',), ('JET_Pile_up_OffsetNPV_1_up',)),
    'JET_PU_PtTerm': (('JET_Pile_up_PtTerm_1_down',), ('JET_Pile_up_PtTerm_1_up',)),
    'JET_PU_Rho': (('JET_Pile_up_RhoTopology_1_down',), ('JET_Pile_up_RhoTopology_1_up',)),
    'JET_Punch': (('JET_PunchThrough_MC15_1_down',), ('JET_PunchThrough_MC15_1_up')),
    'JET_SP_HighPt': (('JET_SingleParticle_HighPt_1_down',), ('JET_SingleParticle_HighPt_1_up',)),
}

SYSTEMATICS_2015 = {
}
SYSTEMATICS_2015.update(SYSTEMATICS_COMMON)

SYSTEMATICS_2016 = {
}
SYSTEMATICS_2016.update(SYSTEMATICS_COMMON)

SYSTEMATICS_BY_WEIGHT = [
#    ('TAU_TRIGGER_SYST_UP',),
#    ('TAU_TRIGGER_SYST_DOWN',),
#    ('TAU_TRIGGER_STATDATA_UP',),
#    ('TAU_TRIGGER_STATDATA_DOWN',),
#    ('TAU_TRIGGER_STATMC_UP',),
#    ('TAU_TRIGGER_STATMC_DOWN',),
#    ('TAU_TRIGGER_UP',)
#    ('TAU_TRIGGER_DOWN',)
    ('TAU_ID_UP',),
    ('TAU_ID_DOWN',),
]


def iter_systematics(include_nominal=False, year=2015, components=None):
    syst = get_systematics(year)
    if include_nominal:
        yield 'NOMINAL'
    terms = components if components is not None else syst.keys()
    for term in terms:
        try:
            variations = syst[term]
        except KeyError:
            raise ValueError("systematic term {0} is not defined".format(term))
        for var in variations:
            yield var


def get_systematics(year=2016):
    if year == 2015:
        return SYSTEMATICS_2015
    elif year == 2016:
        return SYSTEMATICS_2016
    else:
        raise ValueError("No systematics defined for year %d" % year)

def systematic_name(systematic):
    if isinstance(systematic, basestring):
        return systematic
    return '_'.join(systematic)


def parse_systematics(string):
    if not string:
        return None
    return [tuple(token.split('+')) for token in string.split(',')]
