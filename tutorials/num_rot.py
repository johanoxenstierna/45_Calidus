

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, norm
import scipy
from src.m_functions import min_max_normalize_array



pdf = np.log(np.linspace(1, 100, 100))
# pdf += abs(min(pdf))
pdf = min_max_normalize_array(pdf, y_range=[10, 40])

ax0 = plt.plot(pdf, marker='o')

plt.show()





