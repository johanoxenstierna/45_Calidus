
import numpy as np
import P

def jupiter_gi_():

    gi = {}
    gi['r'] = 800  # 5.2
    gi['pi_offset'] = 0.1 * 2 * np.pi
    gi['speed_gi'] = 1.5  #1.5
    gi['tilt'] = 0.05 * 2 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg
    gi['scale'] = 0.4
    gi['centroid_mult'] = 8

    if P.REAL_SCALE:
        gi['scale'] = 0.08
        # gi['scale'] = 0.3
        gi['speed_gi'] = 3.36
        # gi['speed_gi'] = 30.36

    return gi


def everglade_gi_():

    gi = {}
    gi['r'] = 20
    gi['pi_offset'] = 0
    gi['speed_gi'] = 20
    gi['tilt'] = 0.099 * 2 * np.pi
    gi['scale'] = 0.3
    gi['centroid_mult'] = 8

    if P.REAL_SCALE:
        gi['r'] = 5
        gi['scale'] = 0.1
        gi['speed_gi'] = 80

    return gi


def petussia_gi_():

    gi = {}
    gi['r'] = 40
    gi['pi_offset'] = 0
    gi['speed_gi'] = 15
    gi['tilt'] = 0.15 * 2 * np.pi
    gi['scale'] = 0.5
    gi['centroid_mult'] = 8

    if P.REAL_SCALE:
        gi['r'] = 10
        gi['scale'] = 0.2
        gi['speed_gi'] = 60

    return gi


