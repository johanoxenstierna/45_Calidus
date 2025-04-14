import numpy as np
from scipy.stats._multivariate import multivariate_normal

import P as P
from src.gen_trig_fun import *
from src.objects.abstract import AbstractObject, AbstractSSS
from src.wave_funcs import *


class O1C(AbstractObject, AbstractSSS):

    def __init__(_s, o1_id, gi, pics_planet, parent, type):
        AbstractObject.__init__(_s)

        _s.id = o1_id
        _s.gi = gi
        _s.drawn = 1  # begins as drawn
        _s.zorder = None
        _s.type = type
        _s.parent = parent  # parent
        _s.children = []  # added in gen_objects currently
        _s.pic = pics_planet[0]  # the png
        _s.pics_planet = pics_planet
        # _s.axs0o1= []
        _s.centroid = [_s.pic.shape[0] / 2, _s.pic.shape[1] / 2]
        _s.centroids = np.full((P.FRAMES_TOT_BODIES,), fill_value=_s.pic.shape[0] / 2).astype(np.float32)
        aa = 4
        # _s.pic_shadow = None  # below
        # _s.pic = np.copy(_s.pic_planet)  # might be overwritten each frame.
        # _s.pic[:, :, 3] = _s.pic_shadow

        AbstractSSS.__init__(_s)

        # _s.O2 = {}
        _s.moons = {}

        _s.xy = None
        _s.xy_t = None
        _s.vxy = None
        _s.speed_max = None
        _s.zorders = None
        _s.alphas = None
        _s.scale = None
        _s.rotation = None
        _s.peaks = None

        _s.alphas_DL = []
        _s.zorders_DL = []

        _s.set_frame_ss(0, P.FRAMES_TOT_BODIES)  # OBS -1 important cuz arrays are UBE

    def gen_orbit(_s):

        """
        xy_t is now left middle!
        """

        tot_dist = _s.gi['speed_gi'] * P.FRAMES_TOT_BODIES
        num_rot = tot_dist / 6000
        if P.REAL_SCALE:
            pdf = -np.log(np.linspace(1, 100, 100))
            pdf += abs(min(pdf))
            pdf = min_max_normalize_array(pdf, y_range=[10, 40])

        _s.alphas = np.full((P.FRAMES_TOT_BODIES,), fill_value=0)
        _s.zorders = np.full((P.FRAMES_TOT_BODIES,), dtype=int, fill_value=1)

        # Generate the elliptical motion for the planet
        y_squeeze = 0.08
        if _s.id in ['Astro0b', 'Jupiter']:
            y_squeeze = 0.15
        elif _s.id in ['Saturn', 'Uranus', 'Neptune']:
            y_squeeze = 0.2
        _s.xy_t = np.zeros((P.FRAMES_TOT_BODIES, 2), dtype=np.float32)
        _s.xy_t[:, 0] = np.sin(np.linspace(0 + _s.gi['pi_offset'], num_rot * 2 * np.pi + _s.gi['pi_offset'], P.FRAMES_TOT_BODIES)) * _s.gi['r']
        _s.xy_t[:, 1] = -np.cos(np.linspace(0 + _s.gi['pi_offset'], num_rot * 2 * np.pi + _s.gi['pi_offset'], P.FRAMES_TOT_BODIES)) * _s.gi['r'] * y_squeeze

        # Apply tilt by rotating the coordinates ARCTAN2
        cos_theta = np.cos(_s.gi['tilt'])
        sin_theta = np.sin(_s.gi['tilt'])
        x_rot = cos_theta * _s.xy_t[:, 0] - sin_theta * _s.xy_t[:, 1]
        y_rot = sin_theta * _s.xy_t[:, 0] + cos_theta * _s.xy_t[:, 1]

        xy_t_rot = np.copy(_s.xy_t)  # OBS: _s.xy_t no longer rotated!
        xy_t_rot[:, 0] = x_rot
        xy_t_rot[:, 1] = y_rot

        _s.vxy = np.gradient(xy_t_rot, axis=0)  # OBS NOT SHIFTED

        _s.xy = xy_t_rot
        _s.xy[:, 0] += _s.parent.xy[:, 0]
        _s.xy[:, 1] += _s.parent.xy[:, 1]

        inds_neg = np.where(_s.vxy[:, 0] >= 0)[0]  # moving right  OBS ONLY WORKS FOR CLOCKWISE THEN
        # inds_neg = np.where(_s.xy_t[:, 0] < 0)[0]
        _s.zorders[inds_neg] *= -1
        # _s.zorders *= _s.gi['r'] + _s.parent.gi['r']  # + parent radius
        _s.zorders *= _s.gi['r']
        _s.zorders += _s.parent.zorders

        '''OBS has to be done AFTER zorders'''
        _s.vxy[:, 0] += _s.parent.vxy[:, 0]
        _s.vxy[:, 1] += _s.parent.vxy[:, 1]
        _s.speed_max = max(np.linalg.norm(_s.vxy, axis=1))

        aa = 5

        # ONLY DL NOW
        # _s.alphas = np.copy(_s.xy[:, 1])
        # _s.alphas = min_max_normalize_array(_s.alphas, y_range=[0.99, 1])  # will prob be removed in favor of DL

        if _s.parent.id == '0':  # this means the object is rotating around Calidus and not one of its planets (i.e. its not a moon)
            _s.rotation = np.linspace(0 + _s.gi['pi_offset'], num_rot * 2 * np.pi + _s.gi['pi_offset'], P.FRAMES_TOT_BODIES)
            # _s.rotation = np.full((P.FRAMES_TOT_BODIES,), fill_value=0)
            _s.scale = np.copy(_s.xy_t[:, 1])
            _s.scale = min_max_normalize_array(_s.scale, y_range=[0.7 * _s.gi['scale'], _s.gi['scale']])
        else:
            _s.rotation = _s.parent.rotation
            _s.scale = np.full((P.FRAMES_TOT_BODIES,), fill_value=_s.gi['scale'])  # GSS

        _s.centroids *= _s.scale
        # _s.centroids[:, 1] *= _s.scale

        # if _s.id == 'GSS':
        #     _s.scale = np.full((P.FRAMES_TOT_BODIES,), fill_value=0.5)
        #     _s.centroids[:, 0] *= _s.scale
        #     _s.centroids[:, 1] *= _s.scale
        #     adf = 5

    def gen_DL(_s):
        """
        DL: Dark-Light, but its just a convenient 2 letter summary.
        Alphas,
        etc
        """
        '''ALPHAS'''

        y_range_lo = 0.01
        y_range_hi = 0.99
        if _s.id in ['Saturn', 'Uranus', 'Neptune']:
            y_range_hi = 0.2

        #DARK
        if _s.parent.id in ['Nauvis', 'Jupiter']:  # Moons
            alphas0 = np.copy(_s.parent.xy_t[:, 1])  # DARK  # the more down, the more dark
        else:
            alphas0 = np.copy(_s.xy_t[:, 1])  # DARK  # the more down, the more dark
        alphas0 = min_max_normalize_array(alphas0, y_range=[y_range_lo, y_range_hi])  # [0.01, 0.99]
        if _s.id in ['GSS', 'Astro0b']:
            alphas0 = min_max_normalize_array(alphas0, y_range=[y_range_lo, y_range_hi])
        _s.alphas_DL.append(alphas0)

        # MID
        if len(_s.pics_planet) == 3:
            if _s.parent.id == 'Jupiter':
                alphas1 = np.copy(_s.parent.xy_t[:, 1])  #
            else:
                alphas1 = np.copy(_s.xy_t[:, 1])  # does not change value in _s.alphas_DL
            alphas1 = min_max_normalize_array(alphas1, y_range=[0.5 * y_range_hi, y_range_hi])  # ONLY ONE THAT CAN START ABOVE 0.01
            _s.alphas_DL.append(alphas1)

        # LIGHT: uses alphas0
        alphas2 = -np.copy(alphas0)
        # alphas2 /= np.sum(alphas2)
        alphas2 = min_max_normalize_array(alphas2, y_range=[y_range_lo, y_range_hi])  # [0.01, 0.99]
        if _s.id in ['GSS', 'Astro0b']:
            alphas2 = min_max_normalize_array(alphas2, y_range=[0.3 * y_range_hi, y_range_hi])  # [0.01, 0.99]
        _s.alphas_DL.append(alphas2)

        '''ZORDERS'''
        neg_zorders_inds = np.where(_s.xy_t[:, 0] < 0)[0]  # OBS THIS IS FOR DL
        pos_zorders_inds = np.where(_s.xy_t[:, 0] >= 0)[0]

        zorders0 = np.copy(_s.zorders)  # DARK
        zorders0[neg_zorders_inds] -= 1  # when left IS LEFT OF origin, dark starts going behind
        zorders0[pos_zorders_inds] += 1
        _s.zorders_DL.append(zorders0)

        if len(_s.pics_planet) == 3:
            zorders1 = np.copy(_s.zorders) - 2  # always behind
            _s.zorders_DL.append(zorders1)

        zorders2 = np.copy(_s.zorders)  # LIGHT
        zorders2[neg_zorders_inds] += 1
        zorders2[pos_zorders_inds] -= 1
        if _s.id in ['GSS', 'Astro0b']:
            zorders2 += 2  # avoid
        _s.zorders_DL.append(zorders2)

        ff = 5

    def gen_calidus_astro(_s, pi_offset_distr):

        _s.xy = np.zeros((P.FRAMES_TOT_BODIES, 2))
        _s.xy[:, 0] += _s.parent.xy[:, 0] #- _s.centroid[1] // 2
        _s.xy[:, 1] += _s.parent.xy[:, 1] #- _s.centroid[0] // 2

        _s.zorders = np.full((P.FRAMES_TOT_BODIES,), dtype=int, fill_value=_s.gi['zorder'])

        _s.scale = np.full((P.FRAMES_TOT_BODIES,), fill_value=_s.gi['scale'])
        # _s.centroid = [_s.centroid[0] * _s.scale[0], _s.centroid[1] * _s.scale[0]]
        _s.centroids *= _s.scale
        # _s.centroids[:, 1] *= _s.scale

        tot_dist = _s.gi['speed_gi'] * P.FRAMES_TOT_BODIES
        num_alpha = tot_dist / 2000
        num_rot = tot_dist / 4000

        _s.alphas = 0.5 * (np.sin(np.linspace(pi_offset_distr, pi_offset_distr + num_alpha * 2 * np.pi, P.FRAMES_TOT_BODIES)) + 1)
        _s.alphas = min_max_normalize_array(_s.alphas, y_range=[_s.gi['min_alpha'], _s.gi['max_alpha']])

        # _s.rotation = -np.linspace(pi_offset_distr, pi_offset_distr + num_rot * 2 * np.pi, P.FRAMES_TOT_BODIES)
        _s.rotation = -np.linspace(0, num_rot * np.pi, P.FRAMES_TOT_BODIES)

        if _s.id == '0_black':
            _s.rotation = np.full((P.FRAMES_TOT_BODIES,), fill_value=0)
            _s.alphas = np.full((P.FRAMES_TOT_BODIES,), fill_value=1)
        # _s.rotation = np.full((P.FRAMES_TOT_BODIES,), fill_value=0)

    def gen_astro0(_s):

        """
        OBS A LOT OF THIS IS SET DIRECTLY IN ani_helpers
        """

        _s.xy = np.zeros((P.FRAMES_TOT_BODIES, 2))
        _s.xy[:, 0] += _s.parent.xy[:, 0]  # - _s.centroid[1] // 2
        _s.xy[:, 1] += _s.parent.xy[:, 1]  # - _s.centroid[0] // 2

        _s.zorders = np.full((P.FRAMES_TOT_BODIES,), dtype=int, fill_value=_s.gi['zorder'])

        _s.scale = np.full((P.FRAMES_TOT_BODIES,), fill_value=_s.gi['scale'])
        # _s.centroid = [_s.centroid[0] * _s.scale[0], _s.centroid[1] * _s.scale[0]]
        # _s.centroids[:, 0] = 960
        # _s.centroids[:, 1] = 540

        tot_dist = _s.gi['speed_gi'] * P.FRAMES_TOT_BODIES
        num_alpha = tot_dist / 2000
        num_rot = tot_dist / 4000

        _s.alphas = np.full((P.FRAMES_TOT_BODIES,), fill_value=0.99)

        # _s.rotation = -np.linspace(pi_offset_distr, pi_offset_distr + num_rot * 2 * np.pi, P.FRAMES_TOT_BODIES)
        _s.rotation = np.linspace(0, num_rot * np.pi, P.FRAMES_TOT_BODIES)
        # _s.rotation = np.full((P.FRAMES_TOT_BODIES,), fill_value=0)
