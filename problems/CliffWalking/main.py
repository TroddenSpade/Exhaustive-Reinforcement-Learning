from algorithms.TD.QLearning import QLearning
from algorithms.TD.ExpectedSARSA import ExpectedSARSA
from algorithms.TD.SARSA import SARSA
from problems.CliffWalking.CliffWalking import CliffWalking

import matplotlib.pyplot as plt

env = CliffWalking()

ql = QLearning(env)
rew_ql = ql.stats(100, 500)

sarsa = SARSA(env)
rew_sarsa = sarsa.stats(100, 500)

expected = ExpectedSARSA(env)
rew_expected = expected.stats(100, 500)

plt.plot(rew_ql, 'deepskyblue', label="Q-Learning")
plt.plot(rew_sarsa, 'crimson', label="SARSA")
plt.plot(rew_expected, 'lime', label="Expected SARSA")
plt.ylim((-100, 0))
plt.legend()
plt.show()