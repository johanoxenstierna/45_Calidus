
# import cv2
import numpy as np
import random
from copy import deepcopy
import P
from scipy.stats import multivariate_normal
from src.m_functions import min_max_normalization
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt

def decrement_all_index_axs0(index_removed, R):
	"""
	Whenever an axs0 is popped from the list, all index_axs0 with higher index will be wrong and
	need to be decremented by 1.
	"""

	# if o0.index_axs0 != None:
	# 	if o0.index_axs0 > index_removed:
	# 		o0.index_axs0 -= 1

	for rocket in R:
		if rocket.index_axs0 != None:  # if the object was not the one removed
			if rocket.index_axs0 > index_removed:
				rocket.index_axs0 -= 1

		# for o2_key, o2 in o1.O2.items():  # OBS THIS MEANS sps must have same or fewer frames than f
		# 		if o2.index_axs0 != None:
		# 			if o2.index_axs0 > index_removed:
		# 				o2.index_axs0 -= 1

#
# PAINFUL 30 min BUG HERE (something messed up above)
#
# for sp_key, sp in sh.sps.items():
# 	if sp.index_axs0 != None and sp.o1 == None:
# 		if sp.index_axs0 > index_removed:
# 			sp.index_axs0 -= 1


def set_data_O1(o, ax_b):
	"""
	OBS the planets are NEVER REMOVED
	"""

	if o.type == 'body':
		if len(o.pics_planet) > 1:
			for i in range(len(o.axs0_o1)):
				ax0 = o.axs0_o1[i]
				M = mtransforms.Affine2D(). \
						scale(o.scale[o.clock]). \
						rotate_around(o.centroids[o.clock], o.centroids[o.clock], o.rotation[o.clock]). \
						translate(o.xy[o.clock][0], o.xy[o.clock][1]). \
						translate(-o.centroids[o.clock], -o.centroids[o.clock]) + ax_b.transData
				ax0.set_transform(M)
				# ax0.set_alpha(0.1)
				try:
					ax0.set_alpha(o.alphas_DL[i][o.clock])
				except:
					adf = 5
				ax0.set_zorder(int(o.zorders_DL[i][o.clock]))

		else:
			ax0 = o.axs0_o1[0]
			M = mtransforms.Affine2D(). \
				scale(o.scale[o.clock]). \
				rotate_around(o.centroids[o.clock], o.centroids[o.clock], o.rotation[o.clock]). \
				translate(o.xy[o.clock][0], o.xy[o.clock][1]). \
				translate(-o.centroids[o.clock], -o.centroids[o.clock]) + ax_b.transData
			ax0.set_transform(M)
			ax0.set_alpha(o.alphas[o.clock])
			ax0.set_zorder(int(o.zorders[o.clock]))
	elif o.type in ['0_']:
		M = mtransforms.Affine2D(). \
			scale(o.scale[o.clock]). \
			rotate_around(o.centroids[o.clock], o.centroids[o.clock], o.rotation[o.clock]). \
			translate(o.xy[o.clock][0], o.xy[o.clock][1]). \
			translate(-o.centroids[o.clock], -o.centroids[o.clock]) + ax_b.transData
		o.ax0.set_transform(M)
		o.ax0.set_alpha(o.alphas[o.clock])
		o.ax0.set_zorder(int(o.zorders[o.clock]))
	elif o.type in['0_static']:
		pass
		# o.ax0.set_alpha(1)
		# o.ax0.set_zorder(int(o.zorders[o.clock]))

	elif o.type in ['astro']:

		M = mtransforms.Affine2D(). \
				rotate_around(530, 540, o.rotation[o.clock]). \
				scale(1.3, 0.2). \
				skew(0, 0.3). \
				translate(260, 215) + ax_b.transData
		if P.REAL_SCALE:
			M = mtransforms.Affine2D(). \
					rotate_around(540, 540, o.rotation[o.clock]). \
					scale(0.7, 0.1). \
					skew(0, 0.3). \
					translate(580, 375) + ax_b.transData
		o.ax0.set_transform(M)
		o.ax0.set_alpha(o.alphas[o.clock])
		o.ax0.set_zorder(int(o.zorders[o.clock]))


	else:
		raise Exception("This o1 does not exist*&&*(")


def set_data_rocket(o, axs0):
	"""
	OBS centroid shifting is done in rocket.py
	"""
	xys_cur = [[o.xy[o.clock, 0]], [o.xy[o.clock, 1]]]
	# axs0[o.index_axs0].set_data(xys_cur)  # SELECTS A SUBSET OF WHATS ALREADY PLOTTED
	o.ax0.set_data(xys_cur)  # SELECTS A SUBSET OF WHATS ALREADY PLOTTED
	o.ax0.set_alpha(o.alphas[o.clock])
	o.ax0.set_zorder(o.zorders[o.clock])
	# o.ax0.set_linewidth(100)  # not doing anything
	o.ax0.set_color((o.color[o.clock], o.color[o.clock], o.color[o.clock]))

	# plt.setp(axs0[o.index_axs0], markersize=10)





