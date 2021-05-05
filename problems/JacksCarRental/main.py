from algorithms.DP.PolicyIteration import PolicyIteration
from problems.JacksCarRental.JacksRentalCar import JacksRentalCar

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

pe = PolicyIteration(JacksRentalCar, theta=0.1, gamma=0.9)

policy = pe.iteration().reshape(21,21) - 5

print(policy)
ax = sns.heatmap(policy, linewidth=0.5)
plt.show()
