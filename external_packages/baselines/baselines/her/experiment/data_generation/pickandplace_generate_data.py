import gym
import time
import random
import numpy as np
# import rospy
# import roslaunch
#
# from random import randint
# from std_srvs.srv import Empty
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import PoseStamped
# from geometry_msgs.msg import Pose
# from std_msgs.msg import Float64
# from controller_manager_msgs.srv import SwitchController
# from gym.utils import seeding
# from tqdm import tqdm


"""Data generation for the case of a single block with Fetch Arm pick and place"""


def unpack_obs(obs):
    return obs['achieved_goal'], obs['desired_goal'],\
           np.concatenate((obs['observation'], obs['desired_goal'])), \
           np.concatenate((obs['observation'], obs['achieved_goal']))


def compute_success_rate(infos):
    n_demos = len(infos)
    success_rate = 0.0
    if n_demos == 0:
        print('[WARNING] There are no demonstrations')
        return success_rate

    def success_or_fail(info):
        if info[-1]['is_success'] != 0.0:
            return True
        else:
            return False

    if 'is_success' in infos[0][0]:
        for i in range(n_demos):
            success_rate += float(success_or_fail(infos[i]))
    else:
        print('This kind of demonstrations cannot compute success rate!')
        return success_rate

    return success_rate / n_demos


def goToGoal(env, lastObs, use_with_ddpg_her):
    # goal = self.sampleGoal()
    goal = lastObs['desired_goal']

    # objectPosition
    objectPos = lastObs['observation'][3:6]
    gripperPos = lastObs['observation'][:3]
    gripperState = lastObs['observation'][9:11]
    object_rel_pos = lastObs['observation'][6:9]

    print("relative position ", object_rel_pos)
    print("Goal position ", goal)
    print("gripper Position ", gripperPos)
    print("Object Position ", objectPos)
    print("Gripper state  ", gripperState)

    episodeAcs = []
    episodeObs = []
    episodeRews = []
    episodeInfo = []
    episodeDone = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03

    # print("Max episode steps ", env._max_episode_steps)

    timeStep = 0

    if use_with_ddpg_her:
        episodeObs.append(lastObs)
    else:
        _, _, obs, _ = unpack_obs(lastObs)
        episodeObs.append(obs)

    # T: this loop to move gripper near to the object
    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]

        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03  # T: maybe z-axis, we want to move to above object

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i] * 6

        action[len(action) - 1] = 0.05  # T: Open gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        if use_with_ddpg_her:
            episodeObs.append(obsDataNew)
        else:
            _, _, obs, _ = unpack_obs(obsDataNew)
            episodeObs.append(obs)
        episodeAcs.append(action)
        episodeRews.append(reward)
        episodeInfo.append(info)
        episodeDone.append(done)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    # T: this loop move gripper to grasp object
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]

        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i] * 6

        action[len(action) - 1] = -0.005  # T: Close gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        if use_with_ddpg_her:
            episodeObs.append(obsDataNew)
        else:
            _, _, obs, _ = unpack_obs(obsDataNew)
            episodeObs.append(obs)
        episodeAcs.append(action)
        episodeRews.append(reward)
        episodeInfo.append(info)
        episodeDone.append(done)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    # T: This loop move gripper to the goal
    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]

        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i] * 6

        action[len(action) - 1] = -0.005  # T: Close gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        if use_with_ddpg_her:
            episodeObs.append(obsDataNew)
        else:
            _, _, obs, _ = unpack_obs(obsDataNew)
            episodeObs.append(obs)
        episodeAcs.append(action)
        episodeRews.append(reward)
        episodeInfo.append(info)
        episodeDone.append(done)

        objectPos = obsDataNew['observation'][3:6]

    # T: This loop keeps gripper in fixed position and also closes gripper
    while True:
        env.render()
        action = [0, 0, 0, 0]

        action[len(action) - 1] = -0.005  # T: Close gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1
        episodeAcs.append(action)
        episodeRews.append(reward)
        episodeInfo.append(info)
        episodeDone.append(done)

        if use_with_ddpg_her:
            episodeObs.append(obsDataNew)
            if timeStep >= env._max_episode_steps:
                break
        else:
            if timeStep >= env._max_episode_steps:
                break
            else:
                if use_with_ddpg_her:
                    episodeObs.append(obsDataNew)
                else:
                    _, _, obs, _ = unpack_obs(obsDataNew)
                    episodeObs.append(obs)

    print("Total timesteps taken ", timeStep)

    return np.array(episodeObs), np.array(episodeAcs), np.array(episodeRews), sum(episodeRews), \
           episodeInfo, timeStep, np.array(episodeDone)


def main():
    use_with_ddpg_her = True
    env = gym.make('FetchPickAndPlace-v1')
    numItr = 100

    env.reset()
    print("Reset!")
    time.sleep(1)

    ep_returns = []
    actions = []
    observations = []
    rewards = []
    infos = []
    dones = []

    while len(actions) < numItr:
        obs = env.reset()
        # env.render()
        print("Reset!")
        print("ITERATION NUMBER ", len(actions))
        ob, ac, rew, ret, info, time_step, do = goToGoal(env, obs, use_with_ddpg_her)
        if time_step == env._max_episode_steps:
            observations.append(ob)
            actions.append(ac)
            rewards.append(rew)
            ep_returns.append(ret)
            infos.append(info)
            dones.append(do)

    fileName = "demonstration_FetchPickAndPlace"

    success_rate = compute_success_rate(infos)

    print('\nSuccess rate on demonstration: {}'.format(success_rate))
    np.savez_compressed(fileName, ep_rets=ep_returns, obs=observations,
                        rews=rewards, acs=actions, info=infos, done=dones)


if __name__ == "__main__":
    main()
