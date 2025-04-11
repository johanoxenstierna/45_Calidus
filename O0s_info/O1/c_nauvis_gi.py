
import numpy as np

import P

def nauvis_gi_():  # 20

    gi = {}
    gi['r'] = 400
    gi['pi_offset'] = 0.38 * 2 * np.pi
    gi['speed_gi'] = 4  # 4
    gi['tilt'] = 0.1 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg ARCTAN2 MINUS US UP
    gi['scale'] = 0.4  # 0.3
    gi['centroid_mult']= 8

    if P.REAL_SCALE:
        gi['r'] = 154
        gi['scale'] = 0.1
        gi['speed_gi'] = 40

    return gi


def gss_gi_():  # 21
    gi = {}
    gi['r'] = 45 # 25
    gi['pi_offset'] = 0.85 * 2 * np.pi  # 0=middleTop, 0.5=middleBot,
    gi['speed_gi'] = 14.20  #14  # OBS dont use for querying. if its same as parents it means its stationary
    gi['tilt'] = -0.12 * np.pi  # 0.2  ARCTAN2
    gi['scale'] = 0.2  #0.2  # CAREFUL: AFFECTS CENTROID
    gi['centroid_mult'] = 20  #

    return gi


def molli_gi_():

    gi = {}
    gi['r'] = 30
    gi['pi_offset'] = 0
    gi['speed_gi'] = 10
    gi['tilt'] = 0.099 * 2 * np.pi
    gi['scale'] = 0.5
    gi['dist_cond_mult'] = 3

    if P.REAL_SCALE:
        gi['r'] = 10
        gi['scale'] = 0.2
        gi['speed_gi'] = 800

    return gi
