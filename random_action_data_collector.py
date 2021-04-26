import gym
import gym_billiard
# env = gym.make('Curling-v0')
env = gym.make('CurlingCue-v0')
import numpy as np
import matplotlib.pyplot as plt
from math import cos

import time

print('Actions {}'.format(env.action_space))
print("Size of action space: ", env.action_space.shape[0])
print('Obs {}'.format(env.observation_space))
print("Size of obs space: ", env.observation_space.shape[0])

action = env.action_space.sample()
lim = 200

N = 10000
T = 1000

data_in = np.zeros((499*N,6))
data_out = np.zeros((499*N,4))
  
# N episodes
for i in range(N):
  begin_time = time.time()
  # Reset env and local variables
  obs = env.reset()
  
  prev_action = None
  prev_obs = None
  
  # T time steps, but "done" can be attained before T is reached
  for t in range(T):
    if t < lim:
      action = [np.random.uniform(low=0., high=10.), np.random.uniform(low=-np.pi, high=np.pi)]
      
    else:
      action = [np.random.uniform(low=-10., high=10.), np.random.uniform(low=-np.pi, high=np.pi)]
      # env.render(mode='human')

    obs, reward, done, info = env.step(action)

    if done is True:
      pass
    
    if t != 0:

      data_out[i*t] = obs
      data_in[i*t] = np.concatenate((prev_action, prev_obs), axis=None)
    if done:
      end_time = time.time()
      print("{:.1f}".format(i/N*100),"% done", end="\r")
      break

    prev_action = action
    prev_obs = obs


## Output training data
print("data input:\n", data_in)
print(data_in.shape)
data_in_no_0s = data_in[~np.all(data_in == 0, axis=1)]
print("REMOVED 0:\n", data_in_no_0s)
print(data_in_no_0s.shape)

## Input training data
print("data output:\n", data_out)
print(data_out.shape)
data_out_no_0s = data_out[~np.all(data_in == 0, axis=1)] # filter out the same lines as for data_in
print("REMOVED 0:\n", data_out_no_0s)
print(data_out_no_0s.shape)

np.savez("cue_random_action_data_"+str(N)+"_ep", x=data_in_no_0s, y=data_out_no_0s)
