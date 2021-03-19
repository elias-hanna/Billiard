# Created by Elias Hanna from code by Giuseppe Paolo 
# Date: 03/19/2021

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_billiard.utils import physics, parameters
from gym_billiard.envs import billiard_env
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# TODO implement logger

import logging

logger = logging.getLogger(__name__)

class CurlingCue(billiard_env.BilliardEnv):
  """
  State is composed of:
  s = ([ball_x, ball_y, joint0_angle, joint1_angle, joint0_speed, joint1_speed])

  The values that these components can take are:
  ball_x, ball_y -> [-1.5, 1.5]
  joint0_angle -> [-pi/2, pi/2]
  joint1_angle -> [-pi, pi]
  joint0_speed, joint1_speed -> [-50, 50]
  """
  def __init__(self, seed=None, max_steps=500):
    super().__init__(seed, max_steps)
    self.physics_eng = physics.PhysicsSim(use_cue=True)
    self.goals = np.array([[-0.8, .8]])
    self.goalRadius = [0.4]

  def reset(self, desired_ball_pose=None):
    """
    Function to reset the environment.
    - If param RANDOM_BALL_INIT_POSE is set, the ball appears in a random pose, otherwise it will appear at [-0.5, 0.2]
    - If param RANDOM_ARM_INIT_POSE is set, the arm joint positions will be set randomly, otherwise they will have [0, 0]
    :return: Initial observation
    """
    if self.params.RANDOM_BALL_INIT_POSE:
      init_ball_pose = np.array([self.np_random.uniform(low=-1.2, high=1.2),  # x
                                 self.np_random.uniform(low=-1.2, high=1.2)])  # y
    elif desired_ball_pose is not None:
      init_ball_pose = np.array(desired_ball_pose)
    else:
      init_ball_pose = np.array([-0.5, 0.2])

    if self.params.RANDOM_CUE_INIT_ANGLE:
      init_cue_angle = self.np_random.uniform(low= -np.pi, high=np.pi)
    else:
      init_cue_angle = 0

    init_joint_pose = None
    self.physics_eng.reset([init_ball_pose], init_joint_pose, cue_angle=init_cue_angle)
    self.steps = 0
    self.rew_area = None
    return self._get_obs()

  def _get_obs(self):
    """
    This function returns the state after reading the simulator parameters.
    :return: state: composed of ([ball_pose_x, ball_pose_y], [joint0_angle, joint1_angle], [joint0_speed, joint1_speed])
    """
    ball_pose = self.physics_eng.balls[0].position + self.physics_eng.wt_transform
    joint0_t = self.physics_eng.cue['jointW0'].translation
    joint0_v = self.physics_eng.cue['jointW0'].speed
    if np.abs(ball_pose[0]) > 1.5 or np.abs(ball_pose[1]) > 1.5:
      raise ValueError('Ball out of map in position: {}'.format(ball_pose))

    self.state = np.array([ball_pose[0], ball_pose[1], joint0_t, joint0_v])
    return self.state
  
  def reward_function(self, info):
    """
    This function calculates the reward based on the final position of the ball.
    Once the ball is in the reward area, the close is to the center, the higher the reward
    :param info:
    :return:
    """
    if self.steps >= self.params.MAX_ENV_STEPS: # If we are at the end of the episode
      ball_pose = self.state[:2]
      for goal_idx, goal in enumerate(self.goals):
        dist = np.linalg.norm(ball_pose - goal)
        if dist <= self.goalRadius[goal_idx]:
          reward = (self.goalRadius[goal_idx] - dist)/self.goalRadius[goal_idx]
          done = True
          info['rew_area'] = goal_idx
          return reward, done, info
    return 0, False, info

  def step(self, action):
    """
    Performs an environment step.
    :param action: Arm Motor commands. Can be either torques or velocity, according to TORQUE_CONTROL parameter
    :return: state, reward, final, info
    """
    # action = np.clip(action, -1, 1)

    self.steps += 1
    ## Pass motor command
    self.physics_eng.move_joint('jointW0', action[0])
    ## Simulate timestep
    self.physics_eng.step()
    ## Get state
    self._get_obs()
    info = {}

    # Get reward
    reward, done, info = self.reward_function(info)

    if self.steps >= self.params.MAX_ENV_STEPS:  ## Check if max number of steps has been exceeded
      done = True
      info['reason'] = 'Max Steps reached: {}'.format(self.steps)

    return self.state, reward, done, info
