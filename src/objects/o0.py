from src.gen_extent_triangles import *
from src.objects.abstract import AbstractObject
import P as P
import numpy as np
import random


class O0C(AbstractObject):

    def __init__(_s, pic, gi):
        super().__init__()
        _s.id = '0'
        _s.gi = gi
        _s.pic = pic
        _s.O1 = {}

        _s.centroid = [int(pic.shape[0] / 2), int(pic.shape[1] / 2)]

        _s.xy = np.zeros((P.FRAMES_TOT_BODIES, 2), dtype=np.float32)
        _s.vxy = np.zeros((P.FRAMES_TOT_BODIES, 2), dtype=np.float32)
        _s.xy[:, 0] = 960.
        _s.xy[:, 1] = 540.

        _s.xy_t = np.full((P.FRAMES_TOT_BODIES, 2), fill_value=0).astype(np.float32)
        _s.zorders = np.full((P.FRAMES_TOT_BODIES,), fill_value=1000).astype(int)  # these are multiplied with r later
        _s.gi['r'] = 0

