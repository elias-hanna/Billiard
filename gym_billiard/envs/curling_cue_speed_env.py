# Created by Elias Hanna from code by Giuseppe Paolo 
# Date: 03/19/2021

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_billiard.utils import physics, parameters
from gym_billiard.envs import curling_cue_env
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from collections import deque

# TODO implement logger

import logging

logger = logging.getLogger(__name__)

class CurlingCueSpeed(curling_cue_env.CurlingCue):
  """
  State is composed of:
  s = ([ball_x, ball_y, ball_x_speed, ball_y_speed])

  The values that these components can take are:
  ball_x, ball_y -> [-1.5, 1.5]
  ball_x_speed, ball_y_speed -> [-np.inf, np.inf]
  """
  def __init__(self, seed=None, max_steps=500):
    super().__init__(seed, max_steps)

    ## Ball XY positions can be between -1.5 and 1.5
    ## Arm joint can have positons:
    # Joint 0: [-params.CUE_DISTANCE_TO_BALL, params.CUE_DISTANCE_TO_BALL]
    self.observation_space = spaces.Box(low=np.array([-self.params.TABLE_SIZE[0] / 2., -self.params.TABLE_SIZE[1] / 2., # ball pose
                                                      -np.inf, -np.inf]), # ball velocity
                                        high=np.array([self.params.TABLE_SIZE[0] / 2., self.params.TABLE_SIZE[1] / 2., # ball pose
                                                       np.inf, np.inf]), # ball velocity
                                        dtype=np.float32)

  def _get_obs(self):
    """
    This function returns the state after reading the simulator parameters.
    :return: state: composed of ([ball_pose_x, ball_pose_y], [joint0_angle, joint1_angle], [joint0_speed, joint1_speed])
    """
    ball_pose = self.physics_eng.balls[0].position + self.physics_eng.wt_transform
    self.pose_buffer.append(ball_pose)
    ball_velocity = self.pose_buffer[-1] - self.pose_buffer[-2]
    joint0_t = self.physics_eng.cue['jointW0'].translation
    joint0_v = self.physics_eng.cue['jointW0'].speed
    if np.abs(ball_pose[0]) > 1.5 or np.abs(ball_pose[1]) > 1.5:
      raise ValueError('Ball out of map in position: {}'.format(ball_pose))

    self.state = np.array([ball_pose[0], ball_pose[1], ball_velocity[0], ball_velocity[1]])
    return self.state
