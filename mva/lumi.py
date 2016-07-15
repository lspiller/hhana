
from . import NTUPLE_PATH, DEFAULT_STUDENT
from . import log; log = log[__name__]
import os
from glob import glob

LUMI_UNCERT = {
    2010: 0.034,
    2011: 0.018,
    2012: 0.028,
    2015: 0.03,
    2016: 0.03,
}
LUMI = {
    2011: 4523.35,
    2012: 20274.2,
    2015: 3193.68,
    2016: 5118.35,
}

# HACK for random run number splitting
LUMI_TOTAL = {
    2015: (3193.68+5118.35),
    2016: (3193.68+5118.35),
}
# lumi_files = glob(os.path.join(NTUPLE_PATH, DEFAULT_STUDENT, 'lumi_*'))

# for filename in lumi_files:
#     year = int(filename.split('_')[-1])
#     with open(filename, 'r') as f:
#         lumi = float(f.read())
#         LUMI[year] = lumi
#         log.info("Using lumi of %f for %d" % (lumi, year))


def get_lumi_uncert(year):
    return LUMI_UNCERT[year]
