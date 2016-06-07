import math
from rootpy.tree import Cut
from .common import (
    Category_Preselection,
    CUTS_2J, CUTS_VBF, CUTS_BOOSTED)
from .truth import CUTS_TRUE_VBF_CUTBASED, CUTS_TRUE_BOOSTED
# Documentation:
# https://cds.cern.ch/record/1629891/files/ATL-COM-PHYS-2013-1558.pdf

DETA_JETS = Cut('dijet_deta > 2.6')
MASS_JETS = Cut('dijet_vis_mass > 250')

CUTS_VBF_CUTBASED = (
    CUTS_VBF
    & DETA_JETS
    & MASS_JETS
    )

CUTS_BOOSTED_CUTBASED = (
    CUTS_BOOSTED
    )

INF = 1E100

# Cut-based categories
class hh_VBF_LowDR(Category_Preselection):
    name = 'hh_vbf_lowdr'
    label = '#tau_{had}#tau_{had} CB VBF High-p_{T}^{H}'
    latex = '\\textbf{VBF High-$p_T^{H}$}'
    color = 'red'
    jk_number = 7
    linestyle = 'dotted'
    cuts = (
        CUTS_VBF_CUTBASED
        & Cut('ditau_dr < 1.5')
        & Cut('ditau_mmc_maxw_pt > 140'))
    limitbins = {}
    limitbins[2015] = [0, 60, 80, 100, 120, 150, INF] # - new binning
    norm_category = Category_Preselection


class hh_VBF_HighDR_Tight(Category_Preselection):
    name = 'hh_vbf_highdr_tight'
    label = '#tau_{had}#tau_{had} CB VBF Low-p_{T}^{H} Tight'
    latex = '\\textbf{VBF Low-$p_T^{H}$ Tight}'
    jk_number = 9
    color = 'red'
    linestyle = 'verylongdash'
    cuts = (
        CUTS_VBF_CUTBASED
        & (Cut('ditau_dr > 1.5') | Cut('ditau_mmc_maxw_pt < 140'))
        & Cut('dijet_vis_mass > (-250 * dijet_deta + 1550)'))

    limitbins = [0, 70, 100, 125, 150, INF] # - new binning
    norm_category = Category_Preselection


class hh_VBF_HighDR_Loose(Category_Preselection):
    name = 'hh_vbf_highdr_loose'
    label = '#tau_{had}#tau_{had} CB VBF Low-p_{T}^{H} Loose'
    latex = '\\textbf{VBF Low-$p_T^{H}$ Loose}'
    color = 'red'
    linestyle = 'dashed'
    jk_number = 8
    cuts = (
        CUTS_VBF_CUTBASED
        & (Cut('ditau_dr > 1.5') | Cut('ditau_mmc_maxw_pt < 140'))
        & Cut('dijet_vis_mass < (-250 * dijet_deta + 1550)'))
    limitbins = [0, 50, 70, 85, 100, 120, 150, INF] # - new binning
    norm_category = Category_Preselection


class hh_Boosted_Tight(Category_Preselection):
    name = 'hh_boost_tight'
    label = '#tau_{had}#tau_{had} CB Boosted High-p_{T}^{H}'
    latex = '\\textbf{Boosted High-$p_T^{H}$}'
    color = 'blue'
    linestyle = 'verylongdashdot'
    jk_number = 6
    cuts = ((- CUTS_VBF_CUTBASED) & CUTS_BOOSTED_CUTBASED
            & (Cut('ditau_dr < 1.5') & Cut('ditau_mmc_maxw_pt>140')))
    limitbins = {}
    limitbins[2015] = [0, 60, 68, 76, 84, 92, 100, 110, 120, 130, 140, 150, 175, INF] # - new binning
    norm_category = Category_Preselection


class hh_Boosted_Loose(Category_Preselection):
    name = 'hh_boost_loose'
    label = '#tau_{had}#tau_{had} CB Boosted Low-p_{T}^{H}'
    latex = '\\textbf{Boosted Low-$p_T^{H}$}'
    color = 'blue'
    linestyle = 'dotted'
    jk_number = 5
    cuts = ((- CUTS_VBF_CUTBASED) & CUTS_BOOSTED_CUTBASED
            & (Cut('ditau_dr > 1.5') | Cut('ditau_mmc_maxw_pt<140')))
    limitbins = {}
    limitbins[2015] = [0, 70, 100, 110, 125, 150, 200, INF] # - new binning
    norm_category = Category_Preselection

