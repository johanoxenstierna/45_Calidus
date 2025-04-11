
'''In this file, wave steepnessess and extra coordinate shifts are generated'''

import numpy as np
# from scipy.special import dtype

import P
import scipy
from scipy.stats import beta, multivariate_normal, dirichlet
from src.m_functions import min_max_normalization, min_max_normalize_array


def gen_stns():
    """
    New: OBS TH must have a single X index which specifies the particle that is breaking.
    And it breaks every single time!
    """

    PATH_OUT_stns_TZX = './O0s_info/stns_TZX.npy'
    PATH_OUT_TH = './O0s_info/TH.npy'
    stns_TZX = np.zeros(shape=(P.FRAMES_TOT, P.NUM_Z, P.NUM_X), dtype=np.float16)
    TH = np.zeros(shape=(P.FRAMES_TOT, P.NUM_Z, P.NUM_X), dtype=np.uint16)  # 0, 1 and 2
    TS = None  # below

    # '''cov: more second: more x spread. '''
    # sx, sy = 200, 100
    # Scale = np.array([[sx, 0],
    #                   [0, sy]])
    #
    # # Rotation matrix
    # theta = 0.3 * np.pi
    # c, s = np.cos(theta), np.sin(theta)
    # Rot = np.array([[c, -s],
    #                 [s, c]])
    #
    # # Transformation matrix
    # _cov = Scale.dot(Rot)

    C = 4

    FRAMES = 300
    # if P.COMPLEXITY == 1:
    #     FRAMES = P.FRAMES_TOT

    '''steapnesses used by Gerstner'''
    BOUND_LO_y = 2
    BOUND_UP_y = 3
    BOUND_MI_y = 2.5

    if P.NUM_Z < 2:
        PEAK_STEAPN = 1.5
        pdf_ZX = beta.pdf(x=np.linspace(0, 1, P.NUM_X), a=4, b=4, loc=0)
        stns_ZX = min_max_normalize_array(pdf_ZX, y_range=[BOUND_LO_y, BOUND_UP_y])
        peak = scipy.signal.find_peaks(pdf_ZX)[0][0]

        pdf_ZX_post_peak = np.exp(np.linspace(start=-0, stop=-10, num=P.NUM_X - peak))
        stns_ZX_post_peak = min_max_normalize_array(pdf_ZX_post_peak, y_range=[BOUND_LO_y, BOUND_UP_y])
        stns_ZX[peak:] = stns_ZX_post_peak
        # stns_ZX = stns_ZX[::-1]

        stns_reef = np.linspace(start=1.2, stop=0.5, num=P.NUM_X)  # after hitting shore they become less steep
        # stns_ZX *= stns_reef

        TS = gen_TS(pdf_ZX, pdf_ZX_post_peak, peak)  # OBS this is in COORDS!

        H = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.uint16)
        H[0, 0:peak] = 0
        H[0, peak] = 2
        H[0, peak + 1:P.NUM_X] = 1

        stns_TZX[:, 0, :] = stns_ZX
        TH[:, 0, :] = H

    else:
        '''OBS this is not aligned in any sensical way'''
        # rotation_angle = np.linspace(0.95 * np.pi, 1.1 * np.pi, FRAMES)
        rotation_angle = np.linspace(0.9 * np.pi, 0.91 * np.pi, FRAMES)
        # rotation_angle = np.full((FRAMES,), fill_value=1.01 * np.pi)
        # mvns = []

        means = [np.linspace(0.6 * P.NUM_Z, 0.61 * P.NUM_Z, num=FRAMES),
                 np.linspace(0.5 * P.NUM_X, 0.51 * P.NUM_X, num=FRAMES)]  # Z, X
        shifts = [np.linspace(0, 1, num=FRAMES, dtype=int),
                  np.linspace(0, 1, num=FRAMES, dtype=int)]

        C = np.linspace(200, 201, num=FRAMES)  # less=thinner break
        # cov_b_mult = np.linspace(200, 10, num=FRAMES)
        cov_b = 800

        for ii in range(FRAMES):

            if ii % 100 == 0:
                print(ii)

            '''This is mainly there bcs needed to find a good setting not just for where it breaks, 
            but also for how thick it is.
            Mvn will likely be replaced with one exp per segment. One reason that wasnt done from the start
            is that edge values needs sorting out. '''

            angle = rotation_angle[ii]
            # cov_b = C[ii]
            mean = [means[0][ii], means[1][ii]]
            cov = np.array([[cov_b, 0.99 * cov_b],
                            [0.99 * cov_b, cov_b]])  # more const: less spread
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle), np.cos(angle)]])
            rotated_cov = np.dot(np.dot(rotation_matrix, cov), rotation_matrix.T)
            mv = multivariate_normal(mean, rotated_cov)
            # mvns.append(mv.rvs())

            input_z, input_x = np.mgrid[0:P.NUM_Z:1, 0:P.NUM_X:1]
            pos = np.dstack((input_z, input_x))
            stns_ZX = mv.pdf(pos).reshape([P.NUM_Z, P.NUM_X])
            stns_ZX = stns_ZX / np.max(stns_ZX)
            # aaa = shifts[0][ii]
            if P.NUM_Z > 6:
                stns_ZX = np.roll(stns_ZX, shift=shifts[0][ii], axis=1)
            H_Z = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)
            H_X = np.zeros(shape=(P.NUM_Z, P.NUM_X), dtype=np.float16)

            # stns_ZX[int(P.NUM_Z / 2), :] += 0.0001  # make sure theres a peak
            # stns_ZX[:, int(P.NUM_X / 2)] += 0.0001

            # '''
            # STN Z: One stn array per x col.
            # NEW: Require that each x col has a peak which is non first or last
            # '''
            # peak_inds_z = np.zeros((P.NUM_X,), dtype=np.uint16)
            # # stns_ZX[int(P.NUM_Z / 2), :] += 0.0001  # to make sure there is a peak
            # for i in range(P.NUM_X):  # OBS 0 is closest to screen!
            #     peak = np.argmax(stns_ZX[:, i])
            #     # if peak == 0 or peak == len(stns_ZX[:, i]):
            #     #     raise Exception("Require that stn z peaks are not first or last")
            #
            #     peak_inds_z[i] = peak
            #     # stns_ZX[:peak, i] *= np.exp(np.linspace(start=-3.5, stop=0, num=peak))  # everything until peak (from bottom) reduced
            #     stns_ZX[:, i] = min_max_normalize_array(stns_ZX[:, i], y_range=[BOUND_LO_y, BOUND_UP_y])
            #     h_z = np.copy(stns_ZX[:, i])
            #     h_z[:peak] = 0
            #     H_Z[:, i] = h_z

            '''
            STN X: One stn array per z
            '''

            for i in range(P.NUM_Z):  # OBS 0 is closest to screen!
                peak = np.argmax(stns_ZX[i, :])
                stns_ZX[i, peak:] *= np.exp(np.linspace(start=0, stop=-3.5, num=P.NUM_X - peak))  # NO normalization: ONLY SHRINKAGE

                stns_ZX[i, :] = min_max_normalize_array(stns_ZX[i, :], y_range=[BOUND_LO_y, BOUND_UP_y])  # OOOBSS
                h_x = np.copy(stns_ZX[i, :])
                h_x[peak:] = 0
                H_X[i, :] = h_x

            SPLIT_ZX = [0.8, 0.2]

            '''H Z'''
            H = np.zeros((P.NUM_Z, P.NUM_X), dtype=np.uint16)  # fall height for f ONLY f!

            # inds_buildup = np.where((BOUND_LO_y <= H_Z[:, :]) & (H_Z[:, :] <= BOUND_MI_y))
            # inds_break = np.where(BOUND_MI_y < H_Z[:, :])
            # inds_post = np.where(H_Z[:, :] < BOUND_LO_y)
            #
            # H[inds_buildup] += int(100 * SPLIT_ZX[0])
            # H[inds_break] += int(1000 * SPLIT_ZX[0])
            # H[inds_post] += 0

            '''H X'''
            inds_buildup = np.where((BOUND_LO_y <= H_X[:, :]) & (H_X[:, :] <= BOUND_MI_y))
            inds_break = np.where(BOUND_MI_y < H_X[:, :])
            inds_post = np.where(H_X[:, :] < BOUND_LO_y)

            H[inds_buildup] += int(100 * SPLIT_ZX[1])
            H[inds_break] += int(1000 * SPLIT_ZX[1])
            H[inds_post] += 0

            # H[np.where((H > 4) & (H < 500))] = 0
            # H[np.where(H >= 500)] = 2

            H[np.where((H > 4) & (H < 100))] = 1  # build up
            H[np.where(H >= 100)] = 2  # break

            stns_TZX[ii, :, :] = stns_ZX
            TH[ii, :, :] = H

            brkpnt = 4

    np.save(PATH_OUT_stns_TZX, stns_TZX)
    np.save(PATH_OUT_TH, TH)

    return stns_TZX, TH, TS, peak


def gen_TS(pdf, pdf_post_peak, peak):
    """
    Shift through time of real coordinates! Steepness not enough!
    So this needed to make sure the wave rises at middle and is smaller at beg and end.
    Also, there is a function which kills wave after it hits the shore. Its called 'reef'
    and its higher up in this file.
    The coordinates are manually adjusted based on a shift function.
    All of this is just pre-processing hacks to make something look good with minimal effort.
    Difference from earlier is that the start-simple-first principle is applied extremely strictly now.
    """

    TS = np.copy(pdf)  # Shift thourhg time add Y values with this
    TS = min_max_normalize_array(TS, y_range=[0, 1.44]) # 50

    shift_post_peak_pdf = -beta.pdf(x=np.linspace(0, 1, len(pdf_post_peak)), a=2, b=4, loc=0)
    shift_post_peak_pdf = min_max_normalize_array(shift_post_peak_pdf, y_range=[-60, 0])

    TS[peak:] += shift_post_peak_pdf

    return TS


if __name__ == "__main__":
    stns_ZX = gen_stns()

