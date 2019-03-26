import gym
import time
import random
import numpy as np
import rospy
import roslaunch

from random import randint
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding


"""Data generation for the case of a single block with Fetch Arm pick and place"""

ep_returns = []
actions = []
observations = []
rewards = []
infos = []


def main():
    env = gym.make('FetchPickAndPlace-v1')
    numItr = 100
    initStateSpace = "random"

    env.reset()
    print("Reset!")
    time.sleep(1)

    while len(actions) < numItr:
        obs = env.reset()
        #env.render()
        print("Reset!")
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)
        
    fileName = "data_fetch"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"
    
    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos)


def goToGoal(env, lastObs):
    #goal = self.sampleGoal()
    goal = lastObs['desired_goal']

    #objectPosition
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
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03
    
    print("Max episode steps ", env._max_episode_steps)

    timeStep = 0

    episodeObs.append(lastObs)

    # T: this loop to move gripper near to the object
    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]

        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03 # T: maybe z-axis, we want to move to above object

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*6

        action[len(action)-1] = 0.05    # T: Open gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]

    # T: this loop move gripper to grasp object
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]

        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        action[len(action)-1] = -0.005  # T: Close gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]

    # T: This loop move gripper to the goal
    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]

        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6

        action[len(action)-1] = -0.005  # T: Close gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]

    # T: This loop keeps gripper in fixed position and also closes gripper
    while True:
        env.render()
        action = [0, 0, 0, 0]

        action[len(action)-1] = -0.005  # T: Close gripper

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        gripperPos = obsDataNew['observation'][:3]
        gripperState = obsDataNew['observation'][9:11]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= env._max_episode_steps: break

    print("Total timesteps taken ", timeStep)

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()
