
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

