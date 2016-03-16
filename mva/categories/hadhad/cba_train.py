from math import pi
from rootpy.tree import Cut
from .common import (
    Category_Preselection,
    Category_Preselection_DEta_Control,
    CUTS_VBF_CR,
    CUTS_BOOSTED_CR,
    DETA_TAUS)
from .truth import CUTS_TRUE_VBF, CUTS_TRUE_BOOSTED
from ..features import (cuts_vbf, cuts_boosted,
                        fischer_vbf, fischer_boosted)


LEAD_JET_30 = Cut('jet_0_pt > 30')
SUBLEAD_JET_30 = Cut('jet_1_pt > 30')
AT_LEAST_1JET = Cut('jet_0_pt > 30')

CUTS_2J = LEAD_JET_30 & SUBLEAD_JET_30
# VBF category cuts
CUTS_VBF = (
    CUTS_2J
#    & JVT
#    & DETA_TAUS
    )


# Boosted category cuts
CUTS_BOOSTED = (
    AT_LEAST_1JET
    )


class Category_Pre_VBF(Category_Preselection):
    name = 'vbf'
    label = '#tau_{had}#tau_{had} VBF'
    latex = '\\textbf{VBF}'
    color = 'red'
    linestyle = 'dotted'
    jk_number = 6
    common_cuts = Category_Preselection.common_cuts
    cuts = (
        CUTS_VBF
        # & Cut('dEta_jets > 2.0')
        )
    cuts_truth = CUTS_TRUE_VBF
    cut_features = cuts_vbf
    fischer_features = fischer_vbf
    # train with only VBF mode
    signal_train_modes = ['VBF']
    norm_category = Category_Preselection



class Category_Pre_Boosted(Category_Preselection):
    name = 'boosted'
    label = '#tau_{had}#tau_{had} Boosted'
    latex = '\\textbf{Boosted}'
    color = 'blue'
    linestyle = 'dashed'
    jk_number = 5
    common_cuts = Category_Preselection.common_cuts
    cuts = (
#        (- Category_Pre_VBF.cuts)
        CUTS_BOOSTED
        #& Cut(MET_CENTRALITY.format(pi / 6))
        )
    cuts_truth = CUTS_TRUE_BOOSTED
    cut_features = cuts_boosted
    fischer_features = fischer_boosted
    # train with all modes (inherited from Category in base.py)
    #signal_train_modes =
    norm_category = Category_Preselection

