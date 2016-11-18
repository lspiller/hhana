from rootpy.tree import Cut

TAUID = (Cut('n_taus_medium == 2')
          & Cut('n_taus_tight > 0')
          & Cut('ditau_tau0_jet_bdt_medium == 1')
          & Cut('ditau_tau1_jet_bdt_medium == 1'))

ANTI_ID = Cut('n_taus_tight == 0') | Cut('n_taus_medium == 2')

Q = (
    (Cut('ditau_tau0_q == 1') | Cut('ditau_tau0_q == -1'))
    &
    (Cut('ditau_tau1_q == 1') | Cut('ditau_tau1_q == -1'))
    )

OS     = Cut('selection_opposite_sign==1')
NOT_OS = -OS
SS     = Cut('ditau_qxq >=  1')

P1P1 = Cut('ditau_tau0_n_tracks == 1') & Cut('ditau_tau1_n_tracks == 1')
P3P3 = Cut('ditau_tau0_n_tracks == 3') & Cut('ditau_tau1_n_tracks == 3')
P1P3 = (
    (Cut('ditau_tau0_n_tracks == 1') | Cut('ditau_tau0_n_tracks == 3'))
    &
    (Cut('ditau_tau1_n_tracks == 1') | Cut('ditau_tau1_n_tracks == 3')))

TRACK_ISOLATION = (
    Cut('ditau_tau0_n_wide_tracks == 0')
    & # AND
    Cut('ditau_tau1_n_wide_tracks == 0'))

TRACK_NONISOLATION = (
    Cut('ditau_tau0_n_wide_tracks != 0')
    | # OR
    Cut('ditau_tau1_n_wide_tracks != 0'))


REGIONS = {
    'ALL': Cut(),

    'OS': OS & P1P3 & TAUID & Q,
    'OS_ISOL': OS & P1P3 & TAUID & TRACK_ISOLATION & Q,
    'OS_NONISOL': OS & P1P3 & TAUID & TRACK_NONISOLATION & Q,

    'SS': SS & P1P3 & TAUID & Q,
    'SS_ISOL': SS & P1P3 & TAUID & TRACK_ISOLATION & Q,
    'SS_NONISOL': SS & P1P3 & TAUID & TRACK_NONISOLATION & Q,

    'nOS': NOT_OS & TAUID,
    'nOS_ISOL': NOT_OS & TAUID & TRACK_ISOLATION,
    'nOS_NONISOL': NOT_OS & TAUID & TRACK_NONISOLATION,
    'antiIDnOS': NOT_OS & ANTI_ID,

    'OS_NONID': OS & P1P3 & ANTI_ID & Q,

    'NONISOL': TRACK_NONISOLATION,
}

REGION_SYSTEMATICS = {
    'nOS_NONISOL': 'nOS_ISOL',
    'nOS_ISOL': 'nOS_NONISOL',
    #'nOS': ('nOS_ISOL', 'nOS_NONISOL'),
    'nOS': 'antiIDnOS',
}
