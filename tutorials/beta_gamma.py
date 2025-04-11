import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm
import scipy
from src.m_functions import min_max_normalize_array


NUM = 200
pdf = beta.pdf(x=np.arange(0, NUM), loc=10, a=5, b=10, scale=NUM)
pdf = min_max_normalize_array(pdf, y_range=[0, 20])

ax0 = plt.plot(pdf, marker='o')
# ax0 = plt.plot(shift_post_peak_pdf, marker='o')

# beta_rvs = beta.rvs(a=2, b=5, loc=-2, scale=4, size=1500)
# plt.hist(beta_rvs, bins=100)

# _gamma = gamma.pdf(np.linspace(0, 100, 100), 2, 5, 10)
# ax0 = plt.plot(_gamma)

plt.show()





