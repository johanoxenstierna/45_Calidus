

import numpy as np

import P

def mars_gi_():  # 20

    gi = {}
    gi['r'] = 500
    gi['pi_offset'] = 0.1 * 2 * np.pi
    gi['speed_gi'] = 2  # 4
    gi['tilt'] = 0.05 * 2 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg
    gi['scale'] = 0.4
    gi['centroid_mult'] = 4

    if P.REAL_SCALE:
        gi['r'] = 234
        gi['scale'] = 0.1
        gi['speed_gi'] = 21

    return gi