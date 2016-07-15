from rootpy.tree import Cut

__all__ = [
    'get_trigger',
]


TRIG_HH_1 = Cut('HLT_tau35_medium1_tracktwo_tau25_medium1_tracktwo_L1TAU20IM_2TAU12IM == 1')

TRIG_HH_3 = Cut('HLT_tau35_medium1_tracktwo_tau25_medium1_tracktwo == 1')
TRIG_HH_2 = Cut('HLT_tau35_loose1_tracktwo_tau25_medium1_tracktwo_L1TAU20IM_2TAU12IM == 1')
TRIG_HH_15 = TRIG_HH_1 #| TRIG_HH_2

TRIG_HH_16 = TRIG_HH_3 #| TRIG_HH_2
TRIG_LH_1 = Cut('HLT_mu26_imedium == 1')
TRIG_LH_2 = Cut('HLT_e28_lhtight_iloose == 1')
TRIG_LH = TRIG_LH_1 | TRIG_LH_2

def get_trigger(channel='hadhad', year=2015):
    if channel == 'hadhad':
        if year == 2015:
            return TRIG_HH_15
        elif year == 2016:
            return TRIG_HH_16
    elif channel == 'lephad':
        return TRIG_LH
    else:
        raise RuntimeError('wrong channel name')
