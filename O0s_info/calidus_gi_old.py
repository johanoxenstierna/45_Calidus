"""GERSTNER"""

import copy

# from sh_info.shInfoAbstract import ShInfoAbstract
# import scipy.stats
# from scipy.stats import beta, gamma
# import scipy
# from src.trig_functions import min_max_normalization
# from O0s_info.O2.nauvis_gi import _nauvis_info

import P as P
import random
import numpy as np


class CalidusGI:
    """
    The class instance itself is the container for all the info,
    for the parent o0 they are
    """

    def __init__(_s):

        _s.id = 'calidus'
        _s.frame_ss = [0, P.FRAMES_STOP - 50]
        _s.zorder = None
        _s.o1_init_frames = [1]  # ALWAYS
        _s.xy_calidus = [960, 540]
        _s.r = 0



