import numpy as np

import P


def ogun_gi_():

    gi = {}
    gi['r'] = 150
    # gi['pi_offset'] = -2.5
    gi['pi_offset'] = 0.15 * 2 * np.pi
    gi['speed_gi'] = 6  #5
    gi['tilt'] = 0.04 * 2 * np.pi  # 0.3
    gi['scale'] = 0.4
    gi['centroid_mult'] = 8  # this means mid_flight will be cut at 8x radius and landing r will be 4

    if P.REAL_SCALE:
        gi['r'] = 58
        gi['scale'] = 0.2
        gi['speed_gi'] = 166

    return gi