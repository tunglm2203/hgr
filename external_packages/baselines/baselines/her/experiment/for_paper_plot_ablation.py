# DEPRECATED, use baselines.common.plot_util instead

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
                                                                          mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, nargs='+')
parser.add_argument('--smooth', type=int, default=1)
parser.add_argument('--range', type=int, default=-1, help='Number of transitions want to plot')
parser.add_argument('--legend', type=str, default='', nargs='+')
parser.add_argument('--plot_as_epoch', type=bool, default=False)
parser.add_argument('--final_result', type=bool, default=False)
parser.add_argument('--statistic', action='store_true')
parser.add_argument('--emphases', type=int, default=[-1], nargs='+')
parser.add_argument('--ylim', type=float, default=None)
args = parser.parse_args()

# For 5 curves: Vanilla, w/ prioritized goal, w/ prioritized episode, Ours
color_table = ['#1ac938', '#ff7c00', '#8b2be2', '#023eff']

directory = []
for i in range(len(args.dir)):
    if args.dir[i][-1] == '/':
        directory.append(args.dir[i][:-1])
    else:
        directory.append(args.dir[i])
collect_data = []


rc = {'legend.fontsize': 11,
      'axes.titlesize': 14,
      'axes.labelsize': 14,
      'xtick.labelsize': 12,
      'ytick.labelsize': 12,
      'axes.formatter.useoffset': False,
      'axes.formatter.offset_threshold': 1}
plt.rcParams.update(rc)

for dir in directory:
    args.dir = dir
    data = {}
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**',
                                                                                           'progress.csv'))]
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            print('skipping {}'.format(curr_path))
            continue
        print('loading {} ({})'.format(curr_path, len(results['time_step'])))
        with open(os.path.join(curr_path, 'params.json'), 'r') as f:
            params = json.load(f)

        success_rate = np.array(results['test/success_rate'])
        epoch = np.array(results['time_step']) + 1
        env_id = params['env_name']
        replay_strategy = params['replay_strategy']

        if replay_strategy == 'future':
            config = 'her'
        else:
            config = 'ddpg'
        if 'Dense' in env_id:
            config += '-dense'
        else:
            config += '-sparse'
        env_id = env_id.replace('Dense', '')

        # Process and smooth data.
        assert success_rate.shape == epoch.shape
        x = epoch
        y = success_rate
        if args.smooth:
            x, y = smooth_reward_curve(epoch, success_rate)
        assert x.shape == y.shape

        if env_id not in data:
            data[env_id] = {}
        if config not in data[env_id]:
            data[env_id][config] = []
        data[env_id][config].append((x, y))
    collect_data.append(data)

# Plot data.
for i in range(len(collect_data)):
    data = collect_data[i]
    for env_id in sorted(data.keys()):
        print('exporting {}'.format(env_id))

        for config in sorted(data[env_id].keys()):
            xs, ys = zip(*data[env_id][config])
            n_experiments = len(xs)
            if args.range != -1:
                _xs = []
                _ys = []
                for k in range(n_experiments):
                    _xs.append(xs[k][:args.range])
                    _ys.append(ys[k][:args.range])
                xs = _xs
                ys = _ys
            xs, ys = pad(xs), pad(ys)
            assert xs.shape == ys.shape

            if args.statistic:
                if env_id == 'FetchSlide-v1':
                    per = ys.mean(axis=0)
                    _ids = np.where(per >= 0.25)  # 25%
                    _ids = _ids[0][0] if _ids[0].size > 0 else None
                    idx_75 = xs[0][_ids] if _ids is not None else np.inf

                    _ids = np.where(per >= 0.45)  # 50%
                    _ids = _ids[0][0] if _ids[0].size > 0 else None
                    idx_95 = xs[0][_ids] if _ids is not None else np.inf
                    print('{}: 25/50%   @   {}/{}'.format(args.legend[i], idx_75, idx_95))
                else:
                    per = ys.mean(axis=0)
                    _ids = np.where(per>=0.5)  # 50%
                    _ids = _ids[0][0] if _ids[0].size > 0 else None
                    idx_50 = xs[0][_ids] if _ids is not None else np.inf

                    _ids = np.where(per >= 0.75)  # 75%
                    _ids = _ids[0][0] if _ids[0].size > 0 else None
                    idx_75 = xs[0][_ids] if _ids is not None else np.inf

                    _ids = np.where(per >= 0.90)  # 95%
                    _ids = _ids[0][0] if _ids[0].size > 0 else None
                    idx_95 = xs[0][_ids] if _ids is not None else np.inf
                    print('{}: 50/75/95%   @   {}/{}/{}'.format(args.legend[i], idx_50, idx_75, idx_95))

            if args.final_result:
                print("Final success rate, experiment %d: %.2f" % (i, ys[:, -10:].mean() * 100))

            if i in args.emphases:
                if args.plot_as_epoch:
                    xs_data = np.arange(xs[0].shape[0])
                    plt.plot(xs_data, np.nanmedian(ys, axis=0), label=config, color=color_table[i],
                             linewidth=2)
                    plt.fill_between(xs_data, np.nanpercentile(ys, 25, axis=0),
                                     np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color_table[i])
                    plt.xlabel('Epoch')
                else:
                    plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config,color=color_table[i],
                             linewidth=2)
                    plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0),
                                     np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color_table[i])
                    plt.xlabel('Time step')
            else:
                if args.plot_as_epoch:
                    xs_data = np.arange(xs[0].shape[0])
                    plt.plot(xs_data, np.nanmedian(ys, axis=0), label=config, color=color_table[i])
                    plt.fill_between(xs_data, np.nanpercentile(ys, 25, axis=0),
                                     np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color_table[i])
                    plt.xlabel('Epoch')
                else:
                    plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config,color=color_table[i])
                    plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0),
                                     np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=color_table[i])
                    plt.xlabel('Time step')

        plt.title(env_id)
        plt.ylabel('Median Success Rate')
        if args.ylim is not None:
            plt.ylim([0, args.ylim])
        else:
            plt.ylim([0, 1.05])
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

if args.legend != '':
    assert len(args.legend) == len(directory), "Provided legend is not match with number of directories"
    legend_name = args.legend
else:
    legend_name = [directory[i].split('/')[-1] for i in range(len(directory))]
plt.legend(legend_name, loc='lower right')

plt.savefig(os.path.join('fig_ablate_{}.pdf'.format(env_id)), quality=100)
plt.show()
