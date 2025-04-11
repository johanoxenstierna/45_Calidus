import numpy as np

import P

def astro0_gi_():  # 20

    gi = {}
    gi['r'] = 400
    gi['pi_offset'] = 0
    gi['speed_gi'] = 2
    gi['tilt'] = 0  #0.1 * 2 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg
    gi['zorder'] = 1000
    gi['scale'] = 1

    if P.REAL_SCALE:
        gi['r'] = 317
        gi['scale'] = 1
        gi['speed_gi'] = 4

    return gi


def astro0b_gi_():  # 20

    gi = {}
    gi['r'] = 690
    gi['pi_offset'] = 0
    gi['speed_gi'] = 2
    gi['tilt'] = 0.05 * 2 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg
    gi['zorder'] = 1000
    gi['scale'] = 0.25
    gi['centroid_mult'] = 8

    return gi