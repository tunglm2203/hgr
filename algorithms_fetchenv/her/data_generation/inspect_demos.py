import numpy as np
from matplotlib import pyplot as plt
from generate_demo import compute_success_rate
import os

demos_name = 'demonstration_FetchPickAndPlace.npz'
# demos_name = 'demonstration_FetchPush.npz'
# demos_name = 'demonstration_FetchSlide.npz'

data = np.load(demos_name)
ep_return = data['ep_rets']
infos = data['info']
done = data['done']

n_episodes = len(ep_return)

success_rate = compute_success_rate(infos)

print('Number of epsiodes: ', n_episodes)
print('Success rate on demonstration: {}'.format(success_rate))

# Checking demonstration whether is broken?
for i in range(n_episodes):
    if done[i][-1] != 1:
        print('The episode not terminal at: ', i)

# See distribution of successful episode
ep_sort_as_return_idx = np.argsort(ep_return)   # Sorting as increasing
ep_sort_as_return = ep_return[ep_sort_as_return_idx]

# ======= Configure to filter demonstration =======
GET_N_EPISODES = 100
RETURN_THRESH = -10.0
ep_filtered = np.where(ep_sort_as_return > RETURN_THRESH)

plt.figure()
plt.plot(ep_sort_as_return)
plt.title('Return (Undiscounted, sorted)')

plt.figure()
plt.plot(ep_return[ep_sort_as_return_idx[ep_filtered]])
plt.title("Episode have 'return' > %s (%d eps)" % (RETURN_THRESH, len(ep_filtered[0])))

plt.show()

# Filter GET_N_EPISODES best episodes
# Dict of data: ['ep_rets', 'obs', 'rews', 'acs', 'info', 'done']
new_data = {}
for key in data.keys():
    new_data[key] = data[key][-101:-1]

fileName = os.path.join(demos_name.split('.')[0] + '_{}_best'.format(GET_N_EPISODES))
np.savez_compressed(fileName, ep_rets=new_data['ep_rets'], obs=new_data['obs'],
                        rews=new_data['rews'], acs=new_data['acs'], info=new_data['info'], done=new_data['done'])