import matplotlib.pyplot as plt
import numpy as np
import pickle


# file_name = 'td_error_log.pkl.npz'
file_name = 'logs/reaching/td_error/time2/td_error_log.npz'
# f = open(file_name, 'rb')
# d = pickle.load(f)

d = np.load(file_name)
T = 50

freq = d['freq']    # n_episodes x T
td_error = d['td_error']    # # n_episodes x T x max_freq

n_episodes = freq.shape[0]

freq = freq[:, :T-1]

# Plot frequency of all transition
# freq = np.reshape(freq, (1, n_episodes * (T-1)))
# plt.plot(np.arange(n_episodes * (T-1)).reshape(1, n_episodes * (T-1)), freq, 'ro', markersize=3)

ep_idx = 0
trans_idx = 1

# Plot TD-error over all visitation of 1 state
# td_error_1 = td_error[ep_idx, trans_idx, :freq[ep_idx, trans_idx]]
# plt.plot(np.arange(td_error_1.shape[0]), td_error_1, 'ro-', markersize=3)
# print('Frequency: {}, last TD-error: {}'.format(freq[ep_idx, trans_idx], td_error_1[-1]))

# Plot last encountered TD-error over 1 episodes
td_error_1 = np.zeros_like(freq, dtype=np.float)
for i in range(n_episodes):
    for j in range(T-1):
        td_error_1[i, j] = td_error[i, j, int(freq[i, j] - 1)]

td_error_1 = np.reshape(td_error_1, (1, n_episodes * (T-1)))
# plt.plot(np.arange(n_episodes * (T - 1)).reshape(1, n_episodes * (T-1)), td_error_1, 'ro-', markersize=3)
freq_reshape = freq.reshape(1, n_episodes * (T - 1))
max_freq = np.max(freq_reshape)
freq_reshape = freq_reshape / max_freq

plt.bar(np.arange(n_episodes * (T - 1)), freq_reshape.squeeze(), align='center', alpha=1)
plt.bar(np.arange(n_episodes * (T - 1)), td_error_1.squeeze(), align='center', alpha=0.5, color='red')

plt.title('episode={}, index_transition={}'.format(ep_idx, trans_idx))
plt.xlabel('Frequency')
plt.ylabel('TD-error')
plt.show()
