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
total_act = np.array([0., 0.])
for i in range(10):
  obs = env.reset()
  a = env.render(mode='rgb_array')

  for t in range(10000):
    if t < lim:
      action = [10, 5.]
      total_act += np.array(action)

    else:
      action = [0.1, 0.]
    img = env.render()
    env.render(mode='human')
    from matplotlib import pyplot as plt

    plt.imshow(img, interpolation='nearest')
    plt.draw()
    obs, reward, done, info = env.step(action)
    if done:
      break
  print(info)
