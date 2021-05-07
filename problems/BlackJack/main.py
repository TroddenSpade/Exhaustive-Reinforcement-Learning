from algorithms.MC.MCExploringStarts import MCExploringStarts
from algorithms.MC.MCPrediction import MCPrediction

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

env = gym.make('Blackjack-v0')


def policy(state):
    return 0 if state[0] > 19 else 1


mcp = MCPrediction(env, policy)
mcp.iterate(500000)

v0 = mcp.get_value()[12:22, 1:11, 0]
v1 = mcp.get_value()[12:22, 1:11, 1]

x = range(12, 22)
y = range(1, 11)
X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(X, Y, v0, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.00, 1.00)
ax.set_title('No Usable Ace')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, v1, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.00, 1.00)
ax.set_title('Usable Ace')
plt.show()

mces = MCExploringStarts(env)
mces.iterate(1_000_000)

p0 = mces.get_policy()[:, :, 0]
p1 = mces.get_policy()[:, :, 1]

v0 = mces.get_value()[12:22, 1:11, 0]
v1 = mces.get_value()[12:22, 1:11, 1]

x = range(12, 22)
y = range(1, 11)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(X, Y, v0, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
ax.set_zlim(-1.00, 1.00)
ax.set_title('No Usable Ace')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_surface(X, Y, v1, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
ax.set_zlim(-1.00, 1.00)
ax.set_title('Usable Ace')

ax = fig.add_subplot(2, 2, 3)
ax.imshow(p0)
ax.set_xlim(1, 10)
ax.set_ylim(11, 21)
ax.set_yticks(np.arange(11, 22, 1))
ax.set_xticks(np.arange(1, 11, 1))

ax = fig.add_subplot(2, 2, 4)
ax.imshow(p1)
ax.set_xlim(1, 10)
ax.set_ylim(11, 21)
ax.set_yticks(np.arange(11, 22, 1))
ax.set_xticks(np.arange(1, 11, 1))

plt.show()
