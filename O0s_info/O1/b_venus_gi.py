import numpy as np

import P

def venus_gi_():

    gi = {}
    gi['r'] = 200
    # gi['pi_offset'] = -2.5
    gi['pi_offset'] = 0.4 * 2 * np.pi
    gi['speed_gi'] = 6  #5
    gi['tilt'] = 0.04 * 2 * np.pi  # 0.3
    gi['scale'] = 0.44  # 0.33
    gi['centroid_mult'] = 6  # this means mid_flight will be cut at 8x radius and landing r will be 4

    if P.REAL_SCALE == 1:
        gi['r'] = 110
        gi['scale'] = 0.05
        gi['speed_gi'] = 65

    return gi