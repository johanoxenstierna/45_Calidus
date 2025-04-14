



import numpy as np

import P

def saturn_gi_():  # 20

    gi = {}
    gi['r'] = 1000
    gi['pi_offset'] = 0.1 * 2 * np.pi
    gi['speed_gi'] = 0.7  # 4
    gi['tilt'] = 0.07 * 2 * np.pi
    gi['scale'] = 0.5
    gi['centroid_mult'] = 4

    if P.REAL_SCALE:
        gi['r'] = 0
        gi['scale'] = 0
        gi['speed_gi'] = 0

    return gi