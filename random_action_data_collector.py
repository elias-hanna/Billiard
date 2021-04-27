import gym
import gym_billiard
# env = gym.make('Curling-v0')
env = gym.make('CurlingCue-v0')
import numpy as np
import matplotlib.pyplot as plt
from math import cos

print('Actions {}'.format(env.action_space))
print("Size of action space: ", env.action_space.shape[0])
print('Obs {}'.format(env.observation_space))
print("Size of obs space: ", env.observation_space.shape[0])

action = env.action_space.sample()
lim = 200

N = 1000
T = 1000

data_in = np.zeros((T*N,4))
data_out = np.zeros((T*N,2))

tab_cpt = 0
# N episodes
for i in range(N):
  # Reset env and local variables
  obs = env.reset()
  
  prev_action = None
  prev_obs = None
  
  # T time steps, but "done" can be attained before T is reached
  for t in range(T):
    if t == 0:
      action = [np.random.uniform(low=0., high=2.), np.random.uniform(low=-np.pi, high=np.pi)]
      
    obs, reward, done, info = env.step(action)
    
    if t != 0:
      data_out[tab_cpt] = obs
      data_in[tab_cpt] = np.concatenate((prev_action, prev_obs), axis=None)
      tab_cpt += 1
    if done:
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
