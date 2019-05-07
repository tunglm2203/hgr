import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# t = 10
# n = np.random.normal(t, 40, 10000)
# plt.hist(n, bins=30, normed=True, range=(0, 49))
#
# plt.show()


T = 5
batch_size = 1
# future_p = 1. - 1. / (1. + 4.)
future_p = 1.

t_samples = np.random.randint(T, size=batch_size)

her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
future_offset = future_offset.astype(int)
future_t = (t_samples + 1 + future_offset)[her_indexes]


print('t_samples={}, future_t={}, her_indexes={}'.format(t_samples, future_t, her_indexes))