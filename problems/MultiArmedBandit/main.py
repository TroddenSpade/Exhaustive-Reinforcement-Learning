import numpy as np
import matplotlib.pyplot as plt

from algorithms.Bandit.SimpleBandit import SimpleBandit
from algorithms.Bandit.GradientBandit import GradientBandit
from problems.MultiArmedBandit.Bandit import Bandit

n_a = 10
iterations = 200

bandit_sample = Bandit(n_a)

arms = [0] * n_a
for i in range(10):
    arms[i] = np.random.normal(bandit_sample.q_values[i], 1, 2000)  # first problem as a sample

plt.figure(figsize=(12, 8))
plt.ylabel('Rewards distribution')
plt.xlabel('Actions')
plt.xticks(range(1, 11))
plt.yticks(np.arange(-5, 5, 0.5))
plt.violinplot(arms, positions=range(1, 11), showmedians=True)
plt.show(block=False)

rew1 = SimpleBandit(n_actions=10, initial_q=0, policy='epsilon_greedy', epsilon=0.00, steps=1000).stats(2000, Bandit)
rew2 = SimpleBandit(n_actions=10, initial_q=0, policy='epsilon_greedy', epsilon=0.01, steps=1000).stats(2000, Bandit)
rew3 = SimpleBandit(n_actions=10, initial_q=0, policy='epsilon_greedy', epsilon=0.10, steps=1000).stats(2000, Bandit)
rew_ = GradientBandit(n_actions=10, initial_q=4, policy='epsilon_greedy', alpha=0.1, steps=1000).stats(2000, Bandit)

plt.figure(figsize=(12, 6))
plt.plot(rew1, 'g', label='epsilon = 0')
plt.plot(rew2, 'r', label='epsilon = 0.01')
plt.plot(rew3, 'b', label='epsilon = 0.1')
plt.plot(rew_, 'cyan', label='gradient')
plt.legend()
plt.show()

rew3 = SimpleBandit(n_actions=10, initial_q=0, policy='epsilon_greedy', epsilon=0.10, alpha=0.1, steps=1000) \
    .stats(2000, Bandit)
rew4 = SimpleBandit(n_actions=10, initial_q=5, policy='epsilon_greedy', epsilon=0.00, alpha=0.1, steps=1000) \
    .stats(2000, Bandit)

plt.figure(figsize=(12, 6))
plt.yticks(np.arange(0, 3, 0.2))
plt.plot(rew3, 'r', label='Realistic')
plt.plot(rew4, 'b', label='Optimistic')
plt.legend()
plt.show(block=False)

rew5 = SimpleBandit(n_actions=10, initial_q=0, policy='epsilon_greedy', epsilon=0.10, alpha=0.1, steps=1000) \
    .stats(2000, Bandit)
rew6 = SimpleBandit(n_actions=10, initial_q=5, policy='ucb', alpha=0.1, steps=1000) \
    .stats(2000, Bandit)

plt.figure(figsize=(12, 6))
plt.plot(rew5, 'g', label='e-greedy e=0.1')
plt.plot(rew6, 'b', label='ucb c=2')
plt.legend()
plt.show()
