
import os
from scipy.stats._multivariate import multivariate_normal

import matplotlib.pyplot as plt
from src.m_functions import *

import numpy as np

# PATH_IN = './pictures/Calidus0/planets/'
# PATH_OUT = './pictures/Calidus1/planets/'

PATH_IN = './pictures/Calidus0/0_cal/'
PATH_OUT = './pictures/Calidus1/0_cal/'

_, _, file_names = os.walk(PATH_IN).__next__()
# file_names = ['1_Ogun.png']
# file_names = ['2_Venus.png']
# file_names = ['6_EvergladeL.png']
# file_names = ['1_Ogun.png']
# file_names = ['1_Ogun.png']
# file_names = ['6_Jupiter.png']
# file_names = ['3_GSSD.png']
file_names = ['0_sunR.png']

'''
Ogun: 30, min=0.2, max=0.4
Venus: 50: min=0.05, max=0.2
Nauvis: 66: min=0.05, max=0.2  # with atmosphere
Molli: 19: min=0.3, max=0.6
GSS: 20: min=0.3, max=0.6
Jupiter: 88: min=0.01, max=0.1
Everglade: 35: 

20   50     88
0.3  0.05   0.01
0.6  0.2    0.1
'''

for file_name in file_names:

	# if file_name in ['0_sun.png', '0_black.png']:
	# 	continue

	pic = plt.imread(PATH_IN + file_name)
	d = pic.shape[0]
	if 'Ogun' in file_name:
		_min, _max, _c = 0.2, 0.4, 2
	elif 'Venus' in file_name:
		_min, _max, _c = 0.05, 0.2, 2
	elif 'Nauvis' in file_name:
		_min, _max, _c = 0.05, 0.2, 2
	elif 'Molli' in file_name:
		_min, _max, _c = 0.3, 0.6, 2
	elif 'GSS' in file_name:
		_min, _max, _c = 0.3, 0.6, 2
	elif 'Mars' in file_name:
		_min, _max, _c = 0.05, 0.2, 2
	elif 'Jupiter' in file_name:
		_min, _max, _c = 0.05, 0.2, 3
	elif 'Everglade' in file_name:
		_min, _max, _c = 0.1, 0.4, 2
	elif 'Petussia' in file_name:
		_min, _max, _c = 0.3, 0.6, 2
	elif 'Saturn' in file_name:
		_min, _max, _c = 0.05, 0.2, 2
	elif 'Uranus' in file_name:
		_min, _max, _c = 0.05, 0.2, 2
	elif 'Neptune' in file_name:
		_min, _max, _c = 0.05, 0.2, 2
	# else:
	# 	raise Exception("What you're saying to yourself matters.")
	rv_pos = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 15, 0], [0, d * 15]])  # 0_sun more=more visible
	# rv_pos = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 40, 0], [0, d * 40]])  # 0_cal more=more visible
	# rv_pos = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * 2, 0], [0, d * 2]])  # PLANETS more=more visible  GSS: 0.5
	# rv_pos = multivariate_normal(mean=[d / 2, d / 2], cov=[[d * _c, 0], [0, d * _c]])

	x, y = np.mgrid[0:d:1, 0:d:1]
	pos = np.dstack((x, y))

	'''non-sun'''
	# alpha_mask_pos = rv_pos.pdf(pos)
	# alpha_mask_pos = min_max_normalize_array(alpha_mask_pos, y_range=[0.01, 1])
	# alpha_mask_pos = np.clip(alpha_mask_pos, min=_min, max=_max)  # clip bottom  min: more=>less
	# alpha_mask_pos = min_max_normalize_array(alpha_mask_pos, y_range=[0.01, 1])

	'''sun'''
	alpha_mask_pos = rv_pos.pdf(pos)
	alpha_mask_pos = min_max_normalize_array(alpha_mask_pos, y_range=[0.01, 1])
	alpha_mask_pos = np.clip(alpha_mask_pos, min=0.00001, max=0.9)  # clip top
	alpha_mask_pos = np.clip(alpha_mask_pos, min=0.1, max=0.9)  # clip bottom
	# alpha_mask_pos = np.clip(alpha_mask_pos, min=0.00001, max=0.55)  # clip top
	# alpha_mask_pos = np.clip(alpha_mask_pos, min=0.3, max=0.55)  # clip bottom
	# alpha_mask_pos = alpha_mask_pos / np.max(alpha_mask_pos)
	alpha_mask_pos = min_max_normalize_array(alpha_mask_pos, y_range=[0.01, 1])

	adf = 6
	pic[:, :, 3] = alpha_mask_pos
	plt.imsave(PATH_OUT + file_name, pic)

fig, ax = plt.subplots(figsize=(10, 10))
cmap = plt.cm.gist_heat

ax.imshow(alpha_mask_pos, extent=[0, 950, 0, 950], alpha=0.99, cmap=cmap, zorder=1)
plt.show()








# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# x, y = np.mgrid[0:20:1, 0:20:1]
#
# pos = np.dstack((x, y))
#
# rv = multivariate_normal(mean=[9, 9], cov=[[20, 0], [0, 20]])
# # rv = multivariate_normal(mean=0.5, cov=1)
#
# fig0, ax0 = plt.subplots()
# #
# # ax2 = fig2.add_subplot(111)
#
# ax0.contourf(x, y, rv.pdf(pos))
#
# plt.show()
