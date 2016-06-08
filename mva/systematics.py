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
    'TAU_TRIGGER': (('TAU_TRIGGER_SYST_UP',), ('TAU_TRIGGER_SYST_DOWN',)),
    'TAU_RECO': (('TAU_RECO_UP',), ('TAU_RECO_DOWN',)),
    'TAU_ELEOLR': (('TAU_ELEOLR_UP',), ('TAU_ELEOLR_DOWN',)),
    'TAU_ID': (('TAU_ID_UP',), ('TAU_ID_DOWN',)),
    'QCD_FIT': (('QCDFIT_UP',), ('QCDFIT_DOWN',)),
    'Z_FIT': (('ZFIT_UP',), ('ZFIT_DOWN',)),
    'QCD_SHAPE': (('QCDSHAPE_UP',), ('QCDSHAPE_DOWN',)),
}


SYSTEMATICS_2015 = {
# TAU SYSTEMATICS
    'TAU_TRIGGER_STATDATA': (('TAU_TRIGGER_STATDATA_UP',), ('TAU_TRIGGER_STATDATA_DOWN',)),
#    'TAU_TRIGGER_SYST': (('TAU_TRIGGER_SYST_UP',), ('TAU_TRIGGER_SYST_DOWN',)),
    'TAU_TRIGGER_STATMC': (('TAU_TRIGGER_STATMC_UP',), ('TAU_TRIGGER_STATMC_DOWN',)),
    'TAU_TES_DETECTOR': (('TAUS_TRUEHADTAU_SME_TES_DETECTOR_1_up',), ('TAUS_TRUEHADTAU_SME_TES_DETECTOR_1_down',)),
    'TAU_TES_MODEL': (('TAUS_TRUEHADTAU_SME_TES_MODEL_1_up',), ('TAUS_TRUEHADTAU_SME_TES_MODEL_1_down',)),
    'TAU_TES_INSITU': (('TAUS_TRUEHADTAU_SME_TES_INSITU_1_up',), ('TAUS_TRUEHADTAU_SME_TES_INSITU_1_down',)),
# MET SYSTEMATICS
    'MET_SoftTrk_ResoPara': (('MET_SoftTrk_ResoPara',),),
    'MET_SoftTrk_ResoPerp': (('MET_SoftTrk_ResoPerp',),),
    'MET_SoftTrk_Scale': (('MET_SoftTrk_ScaleUp',), ('MET_SoftTrk_ScaleDown',)),
# JET SYSTEMATICS
    'JET_EtaIntercalibration_NonClosure': (('JET_EtaIntercalibration_NonClosure_1_down',), ('JET_EtaIntercalibration_NonClosure_1_up',)),
    'JET_Gro_upedNP_1' : (('JET_Gro_upedNP_1_1_down',), ('JET_Gro_upedNP_1_1_up',)),
    'JET_Gro_upedNP_2' : (('JET_Gro_upedNP_2_1_down',), ('JET_Gro_upedNP_2_1_up',)),
    'JET_Gro_upedNP_3' : (('JET_Gro_upedNP_3_1_down',), ('JET_Gro_upedNP_3_1_up',)),
    'JET_JET_CROSS_CALIB_FORWARD' : (('JET_JET_CROSS_CALIB_FORWARD_1_up',),),
    'JET_JET_NOISE_FORWARD': (('JET_JET_NOISE_FORWARD_1_up',),),
    'JET_JER_NP0': (('JET_JER_NP0_1_down',), ('JET_JER_NP0_1_up',)),
    'JET_JER_NP1': (('JET_JER_NP1_1_down',), ('JET_JER_NP1_1_up',)),
    'JET_JER_NP2': (('JET_JER_NP2_1_down',), ('JET_JER_NP2_1_up',)),
    'JET_JER_NP3': (('JET_JER_NP3_1_down',), ('JET_JER_NP3_1_up',)),
    'JET_JER_NP4': (('JET_JER_NP4_1_down',), ('JET_JER_NP4_1_up',)),
    'JET_JER_NP5': (('JET_JER_NP5_1_down',), ('JET_JER_NP5_1_up',)),
    'JET_JER_NP6': (('JET_JER_NP6_1_down',), ('JET_JER_NP6_1_up',)),
    'JET_JER_NP7': (('JET_JER_NP7_1_down',), ('JET_JER_NP7_1_up',)),
    'JET_JER_NP8': (('JET_JER_NP8_1_down',), ('JET_JER_NP8_1_up',)),
}
SYSTEMATICS_2015.update(SYSTEMATICS_COMMON)

SYSTEMATICS_BY_WEIGHT = [
    ('TAU_TRIGGER_SYST_UP',),
    ('TAU_TRIGGER_SYST_DOWN',),
    ('TAU_TRIGGER_STATDATA_UP',),
    ('TAU_TRIGGER_STATDATA_DOWN',),
    ('TAU_TRIGGER_STATMC_UP',),
    ('TAU_TRIGGER_STATMC_DOWN',),
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


def get_systematics(year=2015):
    if year == 2015:
        return SYSTEMATICS_2015
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
