from rootpy.tree import Cut

__all__ = [
    'get_trigger',
]


#TRIG_HH = Cut('HLT_tau35_medium1_tracktwo_tau25_medium1_tracktwo_L1TAU20IM_2TAU12IM == 1')
TRIG_HH = ( Cut('HLT_tau35_medium1_tracktwo_tau25_medium1_tracktwo == 1')
          & Cut('jet_0_pt > 70.') & Cut('jet_0_eta < 3.2') & Cut('jet_0_eta > -3.2') )

def get_trigger(channel='hadhad', year=1516):
    if channel == 'hadhad':
        return TRIG_HH
    else:
        raise RuntimeError('wrong channel name')
