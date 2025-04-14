



import numpy as np
import P

def neptune_gi_():  # 20

    gi = {}
    gi['r'] = 1150
    gi['pi_offset'] = 0.4 * 2 * np.pi
    gi['speed_gi'] = 0.2  # 4
    gi['tilt'] = 0.07 * 2 * np.pi
    gi['scale'] = 0.4
    gi['centroid_mult'] = 4

    if P.REAL_SCALE:
        gi['r'] = 0
        gi['scale'] = 0
        gi['speed_gi'] = 0

    return gi