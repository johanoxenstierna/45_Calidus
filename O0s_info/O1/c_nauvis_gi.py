
import numpy as np

import P

def nauvis_gi_():  # 20

    gi = {}
    gi['r'] = 400
    gi['pi_offset'] = 0.02 * 2 * np.pi
    gi['speed_gi'] = 4  # 4
    gi['tilt'] = 0.1 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg ARCTAN2 MINUS US UP
    gi['scale'] = 0.4  # 0.3
    gi['centroid_mult']= 4  # ONLY USED TO CONTROL SIZE OF TAKEOFF ORBIT

    if P.REAL_SCALE == 1:
        gi['r'] = 154
        gi['scale'] = 0.05
        gi['speed_gi'] = 40

    return gi


def gss_gi_():  # 21
    gi = {}
    gi['r'] = 25 # 25
    gi['pi_offset'] = 0.2 * 2 * np.pi  # 0=middleTop, 0.5=middleBot,
    gi['speed_gi'] = 10  #14  # OBS dont use for querying. if its same as parents it means its stationary
    gi['tilt'] = 0.12 * np.pi  # 0.2  ARCTAN2
    gi['scale'] = 0.2  #0.2  # CAREFUL: AFFECTS CENTROID
    gi['centroid_mult'] = 10  #  pic=20x20 -> scale=0.2 -> newSize=2px -> *centroid_mult

    return gi


def molli_gi_():

    gi = {}
    gi['r'] = 40
    gi['pi_offset'] = 0
    gi['speed_gi'] = 10
    gi['tilt'] = 0.099 * 2 * np.pi
    gi['scale'] = 0.5
    gi['centroid_mult'] = 4

    if P.REAL_SCALE:
        gi['r'] = 5
        gi['scale'] = 0.2
        gi['speed_gi'] = 200

    return gi
