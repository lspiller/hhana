from rootpy.tree import Cut
from math import pi

from ..base import Category
from ... import MMC_MASS
# All basic cut definitions are here

#TAUID = (
#        (Cut('ditau_tau0_jet_bdt_medium == 1') & \
#         Cut('ditau_tau1_jet_bdt_tight == 1')) | \
#        (Cut('ditau_tau0_jet_bdt_tight == 1') & \
#         Cut('ditau_tau1_jet_bdt_medium == 1'))) & \
#         Cut('n_taus_medium == 2')

TAUID = (Cut('n_taus_medium == 2')
          & Cut('n_taus_tight > 0')
          & Cut('ditau_tau0_jet_bdt_medium == 1')
          & Cut('ditau_tau1_jet_bdt_medium == 1'))

LEAD_TAU_40 = Cut('ditau_tau0_pt > 40')
SUBLEAD_TAU_30 = Cut('ditau_tau1_pt > 30')

LEAD_JET_50 = Cut('jet_0_pt > 50')
SUBLEAD_JET_30 = Cut('jet_1_pt > 30')
AT_LEAST_1JET = Cut('jet_0_pt > 30')

METCENT = Cut('selection_met_centrality==1')
CUTS_2J = LEAD_JET_50 & SUBLEAD_JET_30
CUTS_1J = LEAD_JET_50 & (- SUBLEAD_JET_30)
CUTS_0J = (- LEAD_JET_50)
MET = Cut('met_et > 20')
#MET = Cut('selection_met == 1')
DR_TAUS = Cut('0.8 < ditau_dr < 2.4')
#DR_TAUS = Cut('selection_delta_r == 1')
DETA_TAUS = Cut('ditau_deta < 1.5')
#DETA_TAUS = Cut('selection_delta_eta == 1')
DETA_TAUS_CR = -DETA_TAUS
RESONANCE_PT = Cut('ditau_higgs_pt > 100')
DETA_TAUS_PRESEL = Cut('ditau_deta < 2.0')

JET_TRIG_PT = Cut('jet_0_pt > 70.')
JET_TRIG_ETA = Cut('-3.2 < jet_0_eta') & Cut('jet_0_eta < 3.2')
JET_TRIG = JET_TRIG_ETA & JET_TRIG_PT

LEPTON_VETO = Cut('selection_lepton_veto == 1')# & Cut('ditau_tau0_ele_bdt_loose==0') & Cut('ditau_tau1_ele_bdt_loose==0')
TRIGGER = Cut('selection_trigger == 1')

# use .format() to set centality value
MET_CENTRALITY = 'ditau_met_bisect==1 || (ditau_met_min_dphi < {0})'

GRL = Cut('grl_pass_run_lb == 1')
# common preselection cuts
PRESELECTION = (
    LEAD_TAU_40
    & SUBLEAD_TAU_30
    & MET
    & METCENT
#    & Cut('%s > 0' % MMC_MASS)
    & JET_TRIG
    & DR_TAUS
    & TAUID
    & LEPTON_VETO
    & TRIGGER
    & DETA_TAUS_PRESEL
    )

# VBF category cuts
CUTS_VBF = (
    CUTS_2J
    & DETA_TAUS
    )

CUTS_VBF_CR = (
    CUTS_2J
    & DETA_TAUS_CR
    )

# Boosted category cuts
CUTS_BOOSTED = (
    RESONANCE_PT
    & DETA_TAUS
    )

CUTS_BOOSTED_CR = (
    RESONANCE_PT
    & DETA_TAUS_CR
    )


class Category_Preselection_NO_MET(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = (
    LEAD_TAU_40
    & SUBLEAD_TAU_30
    # & ID_MEDIUM # implemented in regions
    # & MET
    & Cut('%s > 0' % MMC_MASS)
    & DR_TAUS
    # & TAU_SAME_VERTEX
    )

class Category_Preselection_1_JET(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = (
    LEAD_TAU_40
    & SUBLEAD_TAU_30
    & LEAD_JET_50
    # & ID_MEDIUM # implemented in regions
    & MET
    & Cut('%s > 0' % MMC_MASS)
    & DR_TAUS
    # & TAU_SAME_VERTEX
    )

class Category_Preselection(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = (
        PRESELECTION
        & DETA_TAUS
        # & Cut(MET_CENTRALITY.format(pi / 4))
        )

class Category_Prefitselection(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = (
        PRESELECTION
        # & Cut(MET_CENTRALITY.format(pi / 4))
        )
class Category_Loose_Preselection(Category):
    name = 'preselection'
    label = '#tau_{had}#tau_{had} Preselection'
    common_cuts = (
          Cut('ditau_tau0_jet_bdt_medium == 1')
          & Cut('ditau_tau1_jet_bdt_medium == 1')
          & Cut('ditau_met > 20')
          & Cut()
          & DR_TAUS
          & DETA_TAUS
          & LEAD_TAU_40
          & SUBLEAD_TAU_30
        # & Cut(MET_CENTRALITY.format(pi / 4))
        )

class Category_Preselection_DEta_Control(Category_Preselection):
    is_control = True
    name = 'preselection_deta_control'


class Category_1J_Inclusive(Category_Preselection):
    name = '1j_inclusive'
    label = '#tau_{had}#tau_{had} Inclusive 1-Jet'
    common_cuts = Category_Preselection.common_cuts
    cuts = AT_LEAST_1JET
    norm_category = Category_Preselection
