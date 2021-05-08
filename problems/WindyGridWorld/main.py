from problems.WindyGridWorld.WindyGridWorld import WindyGridWorld
from algorithms.TD.SARSA import SARSA

import matplotlib.pyplot as plt
import numpy as np

env = WindyGridWorld()

sarsa = SARSA(env)
rewards = sarsa.estimate(200)

plt.plot(rewards)
plt.show()

print(sarsa.get_value())
print(np.array([env.wind]))