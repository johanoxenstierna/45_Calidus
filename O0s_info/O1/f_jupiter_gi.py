
import numpy as np
import P

def jupiter_gi_():

    gi = {}
    gi['r'] = 1000  # 5.2
    gi['pi_offset'] = 0.22* 2 * np.pi   # 0.2
    gi['speed_gi'] = 1.5  #1.5
    gi['tilt'] = 0.05 * 2 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg
    gi['scale'] = 0.3
    gi['centroid_mult'] = 8

    if P.REAL_SCALE == 1:
        gi['r'] = 800
        gi['pi_offset'] = 0.23 * 2 * np.pi  # 0.2
        gi['scale'] = 0.03
        # gi['scale'] = 0.3
        gi['speed_gi'] = 2.5
        # gi['speed_gi'] = 30.36

    return gi


def everglade_gi_():

    gi = {}
    gi['r'] = 40
    gi['pi_offset'] = 0.2 * 2 * np.pi
    gi['speed_gi'] = 20
    gi['tilt'] = 0.08 * 2 * np.pi
    gi['scale'] = 0.2
    gi['centroid_mult'] = 12

    if P.REAL_SCALE == 1:
        gi['r'] = 5
        gi['scale'] = 0.1
        gi['speed_gi'] = 80

    return gi


def petussia_gi_():

    gi = {}
    gi['r'] = 60
    gi['pi_offset'] = 0
    gi['speed_gi'] = 15
    gi['tilt'] = 0.15 * 2 * np.pi
    gi['scale'] = 0.4
    gi['centroid_mult'] = 12

    if P.REAL_SCALE == 1:
        gi['r'] = 10
        gi['scale'] = 0.2
        gi['speed_gi'] = 60

    return gi


