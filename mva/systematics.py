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
    # FIT PARAMETERS
    'QCD_FIT': (('QCDFIT_UP',), ('QCDFIT_DOWN',)),
    'Z_FIT': (('ZFIT_UP',), ('ZFIT_DOWN',)),
    'QCD_SHAPE': (('QCDSHAPE_UP',), ('QCDSHAPE_DOWN',)),

    # TAU SYS
    'TAU_RECO': (('TAU_RECO_UP',), ('TAU_RECO_DOWN',)),

    # TAU ID
    'TAU_ID': (('TAU_ID_UP',), ('TAU_ID_DOWN',)),

    # TAU TRIGGER
    'TAU_TRIGGER': (('TAU_TRIGGER_DOWN',), ('TAU_TRIGGER_UP',)),
    'TAU_TRIGGER_STATDATA': (('TAU_TRIGGER_STATDATA_UP',), ('TAU_TRIGGER_STATDATA_DOWN',)),
    'TAU_TRIGGER_STATMC': (('TAU_TRIGGER_STATMC_UP',), ('TAU_TRIGGER_STATMC_DOWN',)),

    # TES
    'TAU_TES_DETECTOR': (('TAUS_TRUEHADTAU_SME_TES_DETECTOR_1up',), ('TAUS_TRUEHADTAU_SME_TES_DETECTOR_1down',)),
    'TAU_TES_MODEL': (('TAUS_TRUEHADTAU_SME_TES_MODEL_1up',), ('TAUS_TRUEHADTAU_SME_TES_MODEL_1down',)),
    'TAU_TES_INSITU': (('TAUS_TRUEHADTAU_SME_TES_INSITU_1up',), ('TAUS_TRUEHADTAU_SME_TES_INSITU_1down',)),

    # MET SYSTEMATICS
    'MET_SoftTrk_ResoPara': (('MET_SoftTrk_ResoPara',),),
    'MET_SoftTrk_ResoPerp': (('MET_SoftTrk_ResoPerp',),),
    'MET_SoftTrk_Scale': (('MET_SoftTrk_ScaleUp',), ('MET_SoftTrk_ScaleDown',)),

    # JET SYSTEMATICS
    'JET_BJES_Response': (('JET_BJES_Response_1down',), ('JET_BJES_Response_1up',)),

    'JET_Effective1': (('JET_EffectiveNP_1_1down',), ('JET_EffectiveNP_1_1up',)),
    'JET_Effective2': (('JET_EffectiveNP_2_1down',), ('JET_EffectiveNP_2_1up',)),
    'JET_Effective3': (('JET_EffectiveNP_3_1down',), ('JET_EffectiveNP_3_1up',)),
    'JET_Effective4': (('JET_EffectiveNP_4_1down',), ('JET_EffectiveNP_4_1up',)),
    'JET_Effective5': (('JET_EffectiveNP_5_1down',), ('JET_EffectiveNP_5_1up',)),
    'JET_Effective6': (('JET_EffectiveNP_6restTerm_1down',), ('JET_EffectiveNP_6restTerm_1up',)),

    'JET_Eta_Modelling': (('JET_EtaIntercalibration_Modelling_1down',), ('JET_EtaIntercalibration_Modelling_1up',)),
    'JET_Eta_NonClosure': (('JET_EtaIntercalibration_NonClosure_1down',), ('JET_EtaIntercalibration_NonClosure_1up',)),
    'JET_Eta_Stat': (('JET_EtaIntercalibration_TotalStat_1down',), ('JET_EtaIntercalibration_TotalStat_1up',)),

    'JET_Flavor_Comp': (('JET_Flavor_Composition_1down',), ('JET_Flavor_Composition_1up',)),
    'JET_Flavor_Resp': (('JET_Flavor_Response_1up',), ('JET_Flavor_Response_1down',)),

    'JET_JER': (('JET_JER_SINGLE_NP_1up',),),

    'JET_PU_NPV': (('JET_Pileup_OffsetNPV_1down',), ('JET_Pileup_OffsetNPV_1up',)),
    'JET_PU_MU': (('JET_Pileup_OffsetMu_1down',), ('JET_Pileup_OffsetMu_1up',)),
    'JET_PU_PtTerm': (('JET_Pileup_PtTerm_1down',), ('JET_Pileup_PtTerm_1up',)),
    'JET_PU_Rho': (('JET_Pileup_RhoTopology_1down',), ('JET_Pileup_RhoTopology_1up',)),
    'JET_Punch': (('JET_PunchThrough_MC15_1down',), ('JET_PunchThrough_MC15_1up',)),

    'JET_SP_HighPt': (('JET_SingleParticle_HighPt_1down',), ('JET_SingleParticle_HighPt_1up',)),

    # ELECTRON OVERLAP
    'TAU_ELEOLR': (('TAU_ELEOLR_UP',), ('TAU_ELEOLR_DOWN',)),
}

SYSTEMATICS_2015 = {}
SYSTEMATICS_2015.update(SYSTEMATICS_COMMON)

SYSTEMATICS_2016 = {}
SYSTEMATICS_2016.update(SYSTEMATICS_COMMON)

SYSTEMATICS_1516 = {}
SYSTEMATICS_1516.update(SYSTEMATICS_COMMON)

SYSTEMATICS_BY_WEIGHT = [
    ('TAU_RECO_UP',),
    ('TAU_RECO_DOWN',),
    ('TAU_TRIGGER_STATDATA_UP',),
    ('TAU_TRIGGER_STATDATA_DOWN',),
    ('TAU_TRIGGER_STATMC_UP',),
    ('TAU_TRIGGER_STATMC_DOWN',),
    ('TAU_TRIGGER_UP',),
    ('TAU_TRIGGER_DOWN',),
    ('TAU_ID_UP',),
    ('TAU_ID_DOWN',),
    ('TAU_ELEOLR_UP',),
    ('TAU_ELEOLR_DOWN',),
#    ('TAU_ID_HIGHPT_UP',),
#    ('TAU_ID_HIGHPT_DOWN',),
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
    elif year == 1516:
        return SYSTEMATICS_1516
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
