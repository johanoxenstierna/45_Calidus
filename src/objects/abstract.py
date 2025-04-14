
import numpy as np
import random
from copy import deepcopy
from src.gen_extent_triangles import *
# from projectiles.src.gen_trig_fun import gen_scale_lds
import matplotlib.transforms as mtransforms
import P as P


class AbstractObject:
    """
    This class is supposed to be ridicilously simple
    Obs doesn't have access to the ax itself, only the info about it.
    Obs if an object is to change its movement it needs a new layer instance.
    """

    def __init__(_s):
        _s.drawn = 0  # 0: not drawn, 1: start drawing, 2. continue drawing, 3. end drawing, 4: dynamic flag usage
        _s.clock = 0
        _s.frame_ss = None  # maybe comment
        _s.index_axs0 = None  # OBS USED BY ROCKET
        _s.axs0_inds = []  # OBS ASSUMES OBJS NEVER REMOVED
        _s.axs0_o1 = []
        _s.pic = None
        _s.pics_planet = []
        _s.type = None
        _s.zorder = None

    def set_clock(_s, i):
        """
        The layer classes don't have access to the ax, so
        this essentially tells the ax what to do.
        """

        if i == _s.frame_ss[0]:
            _s.drawn = 1
        elif i > _s.frame_ss[0] and i < _s.frame_ss[1] - 1:
            _s.drawn = 2  # continue. needed bcs ani_update_step will create a new axs0 otherwise
            _s.clock += 1
        elif i == _s.frame_ss[1] - 1:
            _s.drawn = 3  # end drawing
            _s.clock = 99999  # to make sure error will happen if tried to use
        else:  # NEEDED BCS OTHERWISE _s.drawn just stays on 3
            _s.drawn = 0

    def ani_update_step(_s, ax_b, axs0):
        """
        Based on the drawn condition, draw, remove
        If it's drawn, return True (used in animation loop)
        OBS major bug discovered: axs0.pop(index_axs0) OBVIOUSLY results in that all index_axs0 after popped get
        screwed.
        Returns the following index:
        0: don't draw
        1: draw (will result in warp_affine)
        2: ax has just been removed, so decrement all index_axs0

        TODO: _s.drawn and _s.drawBool one of them are clearly redundant.
        """

        # if _s.drawn == 0:  # not drawn, for some reason this is necessary to keep
        #     return 0, None
        if _s.drawn == 1: # start
            # _s.index_axs0 = len(axs0)

            '''This is where picture copy is created'''
            if _s.type in ['body', '0_', '0_static', 'astro']:
                if _s.type == 'body':
                    for pic in _s.pics_planet:
                    # pass
                        axs0.append(ax_b.imshow(pic, zorder=_s.zorder, alpha=0, interpolation='nearest'))
                        _s.axs0_inds.append(len(axs0) - 1)  # THIS -1 IS NEW: Earlier it was done above
                        _s.axs0_o1.append(axs0[-1])
                        # _s.drawn = 2  # this allows pre-filling axs0, but doesnt seem to help

                        # _s.ax0 = axs0[len(axs0) - 1]
                        # _s.ax0 = axs0[len(axs0) - 1]  # - 1
                        # _s.ax0 = axs0[len(axs0) - 1]

                elif _s.type in ['0_', 'astro']:
                    axs0.append(ax_b.imshow(_s.pic, zorder=_s.zorder, alpha=0, interpolation='none'))
                    _s.ax0 = axs0[len(axs0) - 1]
                elif _s.type == '0_static':

                    r = int(_s.pic.shape[0] / 2 * _s.scale)
                    extent = [960 - r, 960 + r,
                              540 + r, 540 - r]

                    axs0.append(ax_b.imshow(_s.pic, zorder=_s.zorder, alpha=_s.alpha, interpolation='none', extent=extent))
                    # axs0.append(ax_b.imshow(_s.pic, extent=extent))

                    # ax_b.imshow(_s.pic, zorder=3000, alpha=0.9, extent=extent)
                    _s.ax0 = axs0[len(axs0) - 1]

            elif _s.type == 'rocket':
                _s.index_axs0 = len(axs0)
                axs0.append(ax_b.plot(_s.xy[0, 0], _s.xy[0, 1], zorder=_s.zorder,
                                     alpha=0, color=(0.99, 0.99, 0.99), marker='o', markersize=1)[0])

                _s.ax0 = axs0[_s.index_axs0]

            else:
                raise Exception("notthing added in ani_update_step")

            return None
        elif _s.drawn == 2:  # continue drawing
            return None
        elif _s.drawn == 3:  # end drawing. OBS ONLY axs0

            if _s.type in ['body', '0_', '0_static', 'astro']:
                return None
            else:
                try:
                    axs0[_s.index_axs0].remove()  # might save CPU-time
                    axs0.pop(_s.index_axs0)  # OBS OBS!!! MAKES axs0 shorter hence all items after index_axs0 now WRONG
                    _s.ax0 = None
                except:
                    raise Exception("ani_update_step CANT REMOVE AX")
                index_removed = _s.index_axs0
                _s.index_axs0 = None  # THIS IS NEEDED BUT NOT SURE WHY
                return index_removed


class AbstractSSS:
    """
    class for all objects that have o0 as parent
    """

    def __init__(_s):
        # _s.occupied = False
        # _s.o0 = sh
        _s.frame_ss = [None, None]
        # _s.id = id
        # _s.pic = pic  # shouldnt be needed here

    def set_frame_ss(_s, ii, num_frames):
        """
        OBS USED BY Smokes and Spl, which are children to ship. Generates frame_ss, scale_ss
        OBS UPDATE: frame_ss reduced by 1 in length to make sure index not exceeded
        """

        # assert(ii + NUM_FRAMES)
        # _s.gi['frame_ss'] = [ii, ii + NUM_FRAMES]    # OVERWRITES
        # _s.frame_ss = _s.gi['frame_ss']  # THIS IS GLOBAL i (hence useless for e.g. ship.extent)
        _s.frame_ss = [ii, ii + num_frames]

    def check_frame_max(_s, ii, NUM_FRAMES):  # defined in specific functions

        exceeds_frame_max = False
        how_many = 0
        if ii + NUM_FRAMES >= P.FRAMES_STOP - 20:
            exceeds_frame_max = True
            how_many = P.FRAMES_STOP - ii - 20
        return exceeds_frame_max, how_many




