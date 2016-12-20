import math


BDT_BLIND = {
    2015: {
        'vbf': 1,
        'boosted': 1},
    2016: {
        'vbf': 1,
        'boosted': 1},
    1516: {
        'vbf': 1,
        'boosted': 1},
    2012: {
        'vbf': 3,
        'boosted': 1},
    2011: {
        'vbf': 3,
        'boosted': 1}
}


def get_label(name, units=True, latex=False):
    info = VARIABLES[name]
    if latex:
        label = info['title']
    else:
        label = info['root']
    if units and 'units' in info:
        label += ' [{0}]'.format(info['units'])
    return label


def get_binning(name, category, year):
    binning = VARIABLES[name]['binning']
    if isinstance(binning, dict):
        if year in binning:
            binning = binning[year]
        if isinstance(binning, dict):
            binning = binning.get(category.name.upper(), binning[None])
    return binning


def get_scale(name):
    info = VARIABLES[name]
    return info.get('scale', 1)


def get_units(name):
    info = VARIABLES[name]
    return info.get('units', None)


def blind_hist(name, hist, year=None, category=None):
    if name.upper() == 'BDT':
        unblind = BDT_BLIND[year][category.name]
        hist[-(unblind + 1):] = (0, 0)
        return
    blind = VARIABLES[name].get('blind', None)
    if blind is None:
        return
    left, right = blind
    left_bin = hist.FindBin(left)
    right_bin = hist.FindBin(right)
    for ibin in xrange(left_bin, right_bin + 1):
        hist[ibin] = (0, 0)


WEIGHTS = {
    'weight_pileup': {
        'title': 'Pile-up Weight',
        'root': 'Pile-up Weight',
        'filename': 'weight_pileup',
        'binning': (50, -.2, 3.)
    },
}

YEAR_VARIABLES = {
    2011: {
        'RunNumber' : {
            'title': r'RunNumber',
            'root': 'Run Number',
            'filename': 'runnumber',
            'binning': [
                177531, 177986, 178163, 179710, 180614,
                182013, 182726, 183544, 185353, 186516,
                186873, 188902, 190503],
        }
    },
    2012: {
        'RunNumber' : {
            'title': r'RunNumber',
            'root': 'Run Number',
            'filename': 'runnumber',
            'binning': [
                200804, 202660, 206248, 207447, 209074, 210184, 211522,
                212619, 213431, 213900, 214281, 215414, 216399],
        }
    }
}

VARIABLES = {

    'n_avg_int_cor': {
        'title': r'$\langle\mu\rangle|_{LB,BCID}$',
        'root': '#font[152]{#LT#mu#GT#cbar}_{LB,BCID}',
        'filename': 'n_avg_int_corr',
        'binning': (40, 0, 40),
        'integer': True,
    },
    'n_avg_int': {
        'title': r'$\langle\mu\rangle|_{LB,BCID}$',
        'root': '#font[152]{#LT#mu#GT#cbar}_{LB,BCID}',
        'filename': 'n_avg_int',
        'binning': (40, 0, 40),
        'integer': True,
    },
    'n_actual_int': {
       'title': r'$\langle\mu\rangle|_{LB}(BCID)$',
       'root': '#font[152]{#LT#mu#GT#cbar}_{LB}#font[52]{(BCID)}',
       'filename': 'n_actual_int',
       'binning': (40, 0, 40),
    },
    'n_jets': {
        'title': r'Number of Selected Jets',
        'root': '#font[52]{Number of Selected Jets}',
        'filename': 'n_jets',
        'binning': (10, -.5, 9.5),
        'integer': True,
    },

    'ditau_deta': {
        'title': r'$\Delta \eta(\tau,\tau)$',
        'root': '#font[152]{#Delta#eta}(#font[152]{#tau},#font[152]{#tau})',
        'filename': 'ditau_deta',
        'binning': {
            'BOOSTED': (10, 0, 1.5),
            'VBF': (10, 0, 1.5),
            'REST': (10, 0, 1.5),
            None: (40, 0, 1.5)},
        'ypadding': (0.5, 0),
    },

    'ditau_vect_sum_pt': {
        'title': r'$\sum \vec{p}_T$',
        'root': '#font[52]{p}_{T}^{Total}',
        'filename': 'ditau_vect_sum_pt',
        'binning': (40, 50, 250),
        'units': 'GeV',
    },
    'ditau_scal_sum_pt': {
        'title': r'$\sum p_T$',
        'root': '#font[152]{#sum} #font[52]{p}_{T}',
        'filename': 'ditau_scal_sum_pt',
        'binning': (40, 50., 250.),
        'units': 'GeV',
    },
    'met_et': {
        'title': r'$E^{miss}_{T}$',
        'root': '#font[52]{E}^{miss}_{T}',
        'filename': 'met_et',
        'binning': {
            'PRESELECTION': (40, 20, 80),
            'REST': (10, 20, 80),
            None: (10, 20, 120)},
        'units': 'GeV',
    },

    'ditau_higgs_pt': {
        'title': r'Higgs $p_{T}$',
        'root': '#font[52]{Higgs} #font[52]{p}_{T}',
        'filename': 'higgs_pt',
        'binning': {
            'PRESELECTION': (40, 0, 300),
            'REST': (10, 50, 180),
            None: (20, 50, 300)},
        'units': 'GeV',
    },
#    'met_etx': {
#        'title': r'$E^{miss}_{T_{x}}$',
#        'root': '#font[52]{E}^{miss}_{T_{x}}',
#        'filename': 'met_etx',
#        'binning': (20, -75, 75),
#        'units': 'GeV',
#        'legend': 'left',
#    },
#    'met_ety': {
#        'title': r'$E^{miss}_{T_{y}}$',
#        'root': '#font[52]{E}^{miss}_{T_{y}}',
#        'filename': 'met_ety',
#        'binning': (20, -75, 75),
#        'units': 'GeV',
#        'legend': 'left',
#    },
#    'met_phi': {
#        'title': r'$E^{miss}_{T} \phi$',
#        'root': '#font[52]{E}^{miss}_{T} #phi',
#        'filename': 'met_phi',
#        'binning': (5, -math.pi, math.pi),
#    },
#    'ditau_met_min_dphi': {
#        'title': r'min[$\Delta\phi$($\tau$,\/$E^{miss}_{T}$)]',
#        'root': '#font[52]{min}[#font[152]{#Delta#phi}(#font[152]{#tau},#font[52]{E}^{miss}_{T})]',
#        'filename': 'ditau_met_min_dphi',
#        'binning': (10, 0, math.pi),
#    },
#    'ditau_met_bisect': {
#        'title': r'$E^{miss}_{T}$ bisects',
#        'root': '#font[52]{E}^{miss}_{T} bisects',
#        'filename': 'ditau_met_bisect',
#        'binning': (2, -0.5, 1.5),
#        'legend': 'left',
#        'integer': True,
#    },

    'ditau_met_centrality': {
        'title': r'$E^{miss}_{T}$ Centrality',
        'root': '#font[52]{E}^{miss}_{T} #font[152]{#phi} centrality',
        'filename': 'tau_met_centrality',
        'binning': (30, 0, math.sqrt(2)),
        'legend': 'left',
    },
#    'tau1_centrality': {
#        'title': r'$#tau_1$ Centrality',
#        'root': '#font[152]{#tau}_{2} #font[152]{#eta} centrality',
#        'filename': 'tau1_centrality',
#        'binning': (10, -math.sqrt(2), math.sqrt(2)),
#        'legend': 'left',
#    },
#    'tau2_centrality': {
#        'title': r'$#tau_2$ Centrality',
#        'root': '#font[52]{#tau_2} #font[152]{#eta Centrality}',
#        'filename': 'tau2_centrality',
#        'binning': (10, -math.sqrt(2), math.sqrt(2)),
#        'legend': 'left',
#    },

    'ditau_vis_mass': {
        'title': r'$m^{vis}_{\tau\tau}$',
        'root': '#font[52]{m}^{vis}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'ditau_vis_mass',
        'binning': {
            'PRESELECTION': (25, 20, 120),
            'REST': (10, 30, 150),
            None: (25, 20, 120)},
        'units': 'GeV',
        'blind': (70, 110),
    },

    'ditau_coll_approx_m': {
        'title': r'$m^{col}_{\tau\tau}$',
        'root': '#font[52]{m}^{col}_{#font[152]{#tau}#font[152]{#tau}}',
        'filename': 'ditau_coll_approx_m',
        'binning': (40, 0, 200),
        'units': 'GeV',
        'blind': (100, 150),
    },
   'ditau_tau0_pt': {
       'title': r'$\tau_{1} p_{T}$',
       'root': '#font[152]{#tau}_{1} #font[52]{p}_{T}',
       'filename': 'ditau_tau0_pt',
       'binning': (30, 30, 150),
       'units': 'GeV',
   },
   'ditau_tau1_pt': {
       'title': r'$\tau_{2} p_{T}$',
       'root': '#font[152]{#tau}_{2} #font[52]{p}_{T}',
       'filename': 'ditau_tau1_pt',
       'binning': (30, 30, 90),
       'units': 'GeV',
   },

   'ditau_tau0_phi': {
       'title': r'$\tau_{1} \phi$',
       'root': '#font[152]{#tau}_{1} #font[152]{#phi}',
       'filename': 'ditau_tau0_phi',
       'binning': (25, -math.pi, math.pi),
       'legend': 'left',
   },
   'ditau_tau1_phi': {
       'title': r'$\tau_{2} \phi$',
       'root': '#font[152]{#tau}_{2} #font[152]{#phi}',
       'filename': 'ditau_tau1_phi',
       'binning': (25, -math.pi, math.pi),
#       'legend': 'left',
   },
   'ditau_tau0_eta': {
       'title': r'$\tau_{1} \eta$',
       'root': '#font[152]{#tau}_{1} #font[152]{#eta}',
       'filename': 'ditau_tau0_eta',
       'binning': (25, -2.5, 2.5),
       'legend': 'left',
   },
   'ditau_tau1_eta': {
       'title': r'$\tau_{2} \eta$',
       'root': '#font[152]{#tau}_{2} #font[152]{#eta}',
       'filename': 'ditau_tau1_eta',
       'binning': (25, -2.5, 2.5),
#       'legend': 'left',
   },
#   'eta_product_jets': {
#       'title': r'$\eta_{jet1} \times \eta_{jet2}$',
#       'root': '#font[152]{#eta}_{j1} #times #font[152]{#eta}_{j2}',
#       'filename': 'eta_product_jets',
#       'binning': (10, -10., 10.),
#       'legend': 'left',
#   },

   'ditau_tau0_n_tracks': {
       'title': r'$\tau_{1}$ Number of Tracks',
       'root': '#font[152]{#tau}_{1} #font[52]{Tracks}',
       'filename': 'ditau_tau0_n_tracks',
       'binning': (5, -.5, 4.5),
       'integer': True,
   },
   'ditau_tau1_n_tracks': {
       'title': r'$\tau_{2}$ Number of Tracks',
       'root': '#font[152]{#tau}_{2} #font[52]{Tracks}',
       'filename': 'ditau_tau1_n_tracks',
       'binning': (5, -.5, 4.5),
       'integer': True,
   },

   'ditau_tau0_jet_bdt_score': {
      'title': r'$\tau_{1}$ BDT Score',
      'root': '#font[152]{#tau}_{1} #font[52]{BDT Score}',
      'filename': 'ditau_tau0_jet_bdt_score',
      'binning': (20, 0.6, 1.),
   },
   'ditau_tau1_jet_bdt_score': {
      'title': r'$\tau_{2}$ BDT Score',
      'root': '#font[152]{#tau}_{2} #font[52]{BDT Score}',
      'filename': 'ditau_tau1_jet_bdt_score',
      'binning': (20, 0.6, 1.),
   },

   'ditau_cosalpha': {
       'title': r'$\cos[\alpha(\tau,\tau)]$',
       'root': '#font[52]{cos}(#font[152]{#alpha}_{#font[152]{#tau}#font[152]{#tau}})',
       'filename': 'ditau_cosalpha',
       'binning': (40, -1., 1.),
       'legend': 'left'
   },
   'ditau_dr': {
       'title': r'$\Delta R(\tau,\tau)$',
       'root': '#font[152]{#Delta}#font[52]{R}(#font[152]{#tau},#font[152]{#tau})',
       'filename': 'ditau_dr',
       'binning': {
           None: (25, 0.6, 2.1),
           'PRESELECTION': (20, 0.6, 2.6),
           },
       'ypadding': (0.5, 0),
   },
   'dijet_dr': {
       'title': r'$\Delta R(j,j)$',
       'root': '#font[152]{#Delta}#font[52]{R}(#font[52]{j},#font[52]{j})',
       'filename': 'dijet_dr',
       'binning': {
           None: (20, 0.0, 6.),
           'PRESELECTION': (30, 0., 6.)},
       'ypadding': (0.5, 0),
   },
   'ditau_dphi': {
       'title': r'$\Delta \phi(\tau,\tau)$',
       'root': '#font[152]{#Delta#phi}(#font[152]{#tau},#font[152]{#tau})',
       'filename': 'ditau_dphi',
       'binning': (20, 0., 2.0),
       'legend': 'left',
   },
   'dijet_deta': {
       'title': r'$\Delta \eta(\j,\j)$',
       'root': '#font[152]{#Delta#eta}(#font[52]{j},#font[52]{j})',
       'filename': 'dEta_jets',
       'binning': {
           'BOOSTED': (20, 0, 1),
           'VBF': (20, 0, 5),
           'REST': (20, 0, 7),
           None: (20, 0, 5)},
       'ypadding': (0.5, 0),
   },
   'ditau_tau0_q': {
      'title': r'$\tau_1$ Charge',
      'root': '#font[152]{#tau}_{1} #font[52]{Charge}',
      'filename': 'ditau_tau0_q',
      'binning': (7, -3.5, 3.5),
      'integer': True,
   },
   'ditau_tau1_q': {
      'title': r'$\tau_2$ Charge',
      'root': '#font[152]{#tau}_{2} #font[52]{Charge}',
      'filename': 'ditau_tau1_q',
      'binning': (7, -3.5, 3.5),
      'integer': True,
   },
   'jet_1_eta': {
       'title': r'jet$_{2}$ $\eta$',
       'root': '#font[152]{#eta}(#font[52]{j}2)',
       'filename': 'jet_1_eta',
       'binning': (20, -5, 5),
       'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED'],
       'legend': 'left',
   },
   'jet_0_eta': {
       'title': r'jet$_{1}$ $\eta$',
       'root': '#font[152]{#eta}(#font[52]{j}1)',
       'filename': 'jet_0_eta',
       'binning': (20, -5, 5),
       'cats': ['2J', 'VBF'],
       'legend': 'left',
   },
   'jet_1_pt': {
       'title': r'jet$_{2}$ $p_{T}$',
       'root': '#font[52]{p}_{T}(#font[52]{j}_2)',
       'filename': 'jet_1_pt',
       'binning': (25, 25, 100),
       'scale': 1,
       'units': 'GeV',
       'cats': ['2J', 'VBF', '1J', '1J_NONBOOSTED', 'PRESELECTION']
   },
   'jet_0_pt': {
       'title': r'jet$_{0}$ $p_{T}$',
       'root': '#font[52]{p}_{T}(#font[52]{j}_1)',
       'filename': 'jet_0_pt',
       'binning': (30, 50, 200),
       'scale': 1,
       'units': 'GeV',
       'cats': ['2J', 'VBF', 'PRESELECTION']
   },
    'ditau_hcm5': {
        'title': r'HCM5',
        'root': '#font[52]{HCM5}',
        'filename': 'HCM5',
        'binning': {
            'BOOSTED': (20, 0.0, 0.01),
            'VBF': (20, 0.0, 0.01),
            'REST': (8, 0.0, 0.01),
            None: (30, 0.0, 0.01)},
        'scale': 0.01,
    },
#    'moment_hcm1': {
#        'title': r'HCM1',
#        'root': '#font[52]{HCM1}',
#        'filename': 'HCM1',
#        'binning': {
#            'BOOSTED': (15, 0.0, 1.0),
#            'VBF': (15, 0.0, 1.0),
#            'REST': (15, 0.0, 1.0),
#            None: (20, 0.0, 1.0)},
#        'scale': 1,
#    },
#    'moment_hcm2': {
#        'title': r'HCM2',
#        'root': '#font[52]{HCM2}',
#        'filename': 'HCM2',
#        'binning': {
#            'BOOSTED': (15, 0.0, 1.0),
#            'VBF': (15, 0.0, 1.0),
#            'REST': (15, 0.0, 1.0),
#            None: (20, 0.0, 1.0)},
#        'scale': 1,
#    },
    'ditau_hcm3': {
        'title': r'HCM3',
        'root': '#font[52]{HCM3}',
        'filename': 'HCM3',
        'binning': {
            'BOOSTED': (20, 0.0, 0.8),
            'VBF': (20, 0.0, 0.8),
            'REST': (20, 0.0, 0.8),
            None: (30, 0.0, 0.8)},
        'scale': 1,
    },

    'dijet_vis_mass': {
        'title': r'$m^{vis}_{jj}$',
        'root': '#font[52]{m}_{#font[52]{j}#font[52]{j}}',
        'filename': 'mass_jet1_jet2',
        'binning': (25, 0, 500),
        'units': 'GeV',
    },
    'ditau_pt_ratio': {
        'title': r'$\tau_{2} p_{T} / \tau_{1} p_{T}$',
        'root': '#font[52]{p}_{T}(#font[152]{#tau}_{2}) / #font[52]{p}_{T}(#font[152]{#tau}_{1})',
        'filename': 'ditau_pt_ratio',
        'binning': (20, 0.2, 1.),
    },
}

from . import MMC_MASS

VARIABLES[MMC_MASS] = {
    'title': r'$m^{MMC}_{\tau\tau}$',
    'root': '#font[52]{m}^{MMC}_{#font[152]{#tau}#font[152]{#tau}}',
    'filename': MMC_MASS,
    'binning': {
        2015: (24, 40, 160),
        2016: (24, 40, 160),
        1516: (40, 0, 200)},
    'units': 'GeV',
    'blind': (100, 150),
}
