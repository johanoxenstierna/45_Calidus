
import os
import random
import numpy as np

from scipy.stats._multivariate import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from src.m_functions import *


def gen_astro_alpha(d):
    '''Alpha mask'''



    '''astro'''
    rv_pos = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 73, 0], [0, d * 73]])  # more=more visible
    rv_neg = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 65, 0], [0, d * 65]])  # more=less visible

    '''0_cal'''
    # rv_pos = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 50, 0], [0, d * 50]])  # more=more visible
    # rv_neg = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 15, 0], [0, d * 15]])  # more=less visible

    x, y = np.mgrid[0:d:1, 0:d:1]
    pos = np.dstack((x, y))

    # '''o_cal'''
    # alpha_mask_pos = rv_pos.pdf(pos)
    # alpha_mask_pos = min_max_normalize_array(alpha_mask_pos, y_range=[0.01, 1])
    # alpha_mask_pos = np.clip(alpha_mask_pos, min=0.00001, max=0.2)  # clip top    0.1 astro
    # alpha_mask_pos = np.clip(alpha_mask_pos, min=0.1, max=0.2)  # clip bottom  min: more=>less
    # alpha_mask_pos = alpha_mask_pos / np.max(alpha_mask_pos)
    #
    # alpha_mask_neg = rv_neg.pdf(pos)
    # alpha_mask_neg = min_max_normalize_array(alpha_mask_neg, y_range=[0.01, 1])
    # alpha_mask_neg = np.clip(alpha_mask_neg, min=0.00001, max=0.2)  # clip top
    # alpha_mask_neg = np.clip(alpha_mask_neg, min=0.1, max=0.2)  # clip bottom
    # alpha_mask_neg = -alpha_mask_neg / np.max(alpha_mask_neg)

    '''astro'''
    alpha_mask_pos = rv_pos.pdf(pos)
    alpha_mask_pos = min_max_normalize_array(alpha_mask_pos, y_range=[0.01, 1])
    alpha_mask_pos = np.clip(alpha_mask_pos, min=0.17, max=0.2)  # clip top    0.1 astro  THE CLOSER THE TIGHTER
    # alpha_mask_pos = np.clip(alpha_mask_pos, min=0.15, max=0.2)  # PEND DEL lip bottom  min: more=>less
    alpha_mask_pos = alpha_mask_pos / np.max(alpha_mask_pos)

    alpha_mask_neg = rv_neg.pdf(pos)
    alpha_mask_neg = min_max_normalize_array(alpha_mask_neg, y_range=[0.01, 1])
    alpha_mask_neg = np.clip(alpha_mask_neg, min=0.17, max=0.2)  # clip top
    # alpha_mask_neg = np.clip(alpha_mask_neg, min=0.15, max=0.2)  # PEND DEL
    alpha_mask_neg = -alpha_mask_neg / np.max(alpha_mask_neg)

    alpha_mask = 0.5 * alpha_mask_pos + 0.5 * alpha_mask_neg

    alpha_mask = min_max_normalize_array(alpha_mask, y_range=[0, 1])

    return alpha_mask, x, y


PATH_IN = './pictures/Calidus0/Astro0.png'
PATH_OUT = './pictures/Calidus1/z_Astro0_masked.png'

# PATH_IN = './pictures/Calidus0/0_cal/0h_light.png'
# PATH_OUT = './pictures/Calidus1/0_cal/0h_light.png'

d = 1080 #370  # 1080

alpha_mask, x, y = gen_astro_alpha(d)
fig, ax = plt.subplots(figsize=(10, 10))
# alpha_mask_pdf_n = alpha_mask / np.max(alpha_mask)
# ax.contourf(x, y, alpha_mask)

astro0 = imread(PATH_IN)
astro0_masked = np.copy(astro0)
# alpha_mask = np.copy(astro0[:, :, 3])
astro0_masked[:, :, 3] *= 1 * alpha_mask  # needs to be combination
astro0_masked = min_max_normalize_array(astro0_masked, y_range=[0, 1])
# plt.imsave(PATH_OUT + file_name, pic)

cmap = plt.cm.gist_heat

ax.imshow(alpha_mask, extent=[0, d, 0, d], alpha=0.99, cmap=cmap, zorder=1)
# ax.imshow(astro0_masked, extent=[0, d, 0, d], alpha=0.99, zorder=1)
plt.imsave(PATH_OUT, astro0_masked)

plt.show()


# def cut_k0(k0, inds_x, inds_z, d=None):
#     """
#     Generates the static pics
#     """
#
#     PATH_OUT = './pictures/k0_cut/'
#
#     delete_old(PATH_OUT)
#
#     '''Alpha mask'''
#     # rv = multivariate_normal(mean=[d/2, d/2], cov=[[d*4, 0], [0, d*4]])  # more=more visible
#     rv = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 40000, 0], [0, d * 40000]])  # more=more visible
#     # x, y = np.mgrid[0:d*2:1, 0:d*2:1]
#     x, y = np.mgrid[0:d:1, 0:d:1]
#     pos = np.dstack((x, y))
#     alpha_mask = rv.pdf(pos)
#     alpha_mask = alpha_mask / np.max(alpha_mask)
#
#     '''
#     Obs this needs to correspond exactly with k0.
#     Needs to be flipped somehow.
#     plt.gca() is FLIPPED
#     Caution this is fk up.
#     '''
#
#     for i in range(len(inds_x)):
#         for j in range(len(inds_z)):
#             ind_x = inds_x[i]
#             ind_z = inds_z[j]
#
#             if ind_z < int(d / 2):
#                 print("ind_z: " + str(ind_z), "   d: " + str(d))
#                 raise Exception("d too large")
#
#             # pic = k0[ind_z + int(d/2):ind_z - int(d/2):-1, ind_x - int(d/2):ind_x + int(d/2), :]
#             pic = k0[ind_z - int(d / 2):ind_z + int(d / 2), ind_x - int(d / 2):ind_x + int(d / 2), :]
#
#             pic[:, :, 3] = alpha_mask
#
#             pic_key = str(ind_x) + '_' + str(ind_z)
#             np.save(PATH_OUT + pic_key, pic)
#
#
# def delete_old(PATH):
#     _, _, all_file_names = os.walk(PATH).__next__()
#
#     removed_files = 0
#     for file_name_rem in all_file_names:
#         # print("removing " + str(file_name_rem))
#         os.remove(PATH + file_name_rem)
#         removed_files += 1
#     print("removed_files: " + str(removed_files))
#
#
# def get_c_d(k0, d):
#     """
#     How to sample the mini-images based on where they are sampled in k0
#     """
#     # c_ = k0[720:719 - d:-1, 100:100 + d, :]
#     c_ = k0[720:719 - d:-1, 100:100 + d * 2, :]
#     # d_ = k0[720:719 - d:-1, 0:d, :]
#     d_ = k0[720:719 - d:-1, 0:d * 2, :]
#
#     # rv = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 3, 0], [0, d * 3]])  # less cov => less alpha, second one: width
#     rv = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 3, 0], [0, d * 6]])
#     # x, y = np.mgrid[0:d*2:1, 0:d*2:1]
#     x, y = np.mgrid[0:d:1, 0:d * 2:1]
#     pos = np.dstack((x, y))
#     mask = rv.pdf(pos)
#     mask = mask / np.max(mask)
#
#     c_[:, :, 3] = mask
#     d_[:, :, 3] = mask
#
#     return c_, d_
#
#
# def convert_pixels_to_possible_cells(pixels):
#     cells = []
#
#     '''Get ratio'''
#     pxsl_x = int(pixels.split('_')[0])
#     pxsl_z = int(pixels.split('_')[1])
#
#     ratio_x = pxsl_x / 1280
#     ratio_z = (720 - pxsl_z) / 720  # FROM BOTTOM
#
#     ratio_x_diff = 0.3
#     ratio_z_diff = 0.3
#
#     cell_x_min = int((ratio_x - ratio_x_diff) * P.NUM_X)
#     cell_x_max = int((ratio_x + ratio_x_diff) * P.NUM_X)
#
#     cell_z_min = int((ratio_z - ratio_z_diff) * P.NUM_Z)
#     cell_z_max = int((ratio_z + ratio_z_diff) * P.NUM_Z)
#
#     for i in range(cell_x_min, cell_x_max):
#         for j in range(cell_z_min, cell_z_max):
#             cells.append((i, j))
#
#     return cells