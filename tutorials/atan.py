

import numpy as np

vxy_i = np.array([20, -80])
# desired_vxy = np.array([90, -40])

# vxy_i = np.array([-40, 8])

vxy_i_theta = np.arctan(vxy_i[1] / vxy_i[0])
# desired_vxy_theta = np.arctan2(desired_vxy[1], desired_vxy[0])

aa = np.rad2deg(vxy_i_theta)
# bb = np.rad2deg(desired_vxy_theta)

# angle_diff = np.arctan2(np.sin(desired_vxy_theta - vxy_i_theta), np.cos(desired_vxy_theta - vxy_i_theta))

adf = 5
