import math
from rootpy.tree import Cut
from .common import (
    Category_Preselection,
    CUTS_2J, CUTS_VBF, CUTS_BOOSTED)
from .truth import CUTS_TRUE_VBF_CUTBASED, CUTS_TRUE_BOOSTED
# Documentation:
# https://cds.cern.ch/record/1629891/files/ATL-COM-PHYS-2013-1558.pdf

OPTICUTS_BOOSTED = (
        Cut('ditau_mmc_maxw_pt > 110')
        & Cut('ditau_dr < 1.6')
        & Cut('ditau_deta < 1.6')
        )

OPTICUTS_VBF = (
        Cut('ditau_dr < 1.6')
        & Cut('ditau_deta < 1.55')
        & Cut('dijet_deta > 3')
        & Cut('jet_0_pt > 60')
        & Cut('jet_1_pt > 50')
        )


INF = 1E100

class Category_OptiCuts_VBF(Category_Preselection):
    name = 'opticuts_vbf'
    label = '#tau_{had}#tau_{had} CB VBF'
    latex = '\\textbf{VBF}'
    jk_number = 9
    color = 'red'
    linestyle = 'verylongdash'
    cuts = (
        OPTICUTS_VBF
        )

    limitbins = [0, 70, 100, 115, 135, 150, INF]
    norm_category = Category_Preselection


class Category_OptiCuts_Boosted(Category_Preselection):
    name = 'opticuts_boosted'
    label = '#tau_{had}#tau_{had} CB Boosted'
    latex = '\\textbf{Boosted$}'
    color = 'blue'
    linestyle = 'verylongdashdot'
    jk_number = 6
    cuts = ((- OPTICUTS_VBF) & OPTICUTS_BOOSTED)

    limitbins = {}
    limitbins[2015] = [0, 68, 76, 84, 92, 100, 150, INF]
    norm_category = Category_Preselection
