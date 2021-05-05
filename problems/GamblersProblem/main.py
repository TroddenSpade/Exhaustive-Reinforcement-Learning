from algorithms.DP.ValueIteration import ValueIteration
from problems.GamblersProblem.GamblersProblem import GamblersProblem

import matplotlib.pyplot as plt
import numpy as np

vi = ValueIteration(GamblersProblem, 0.0001, 1)
vi.iterate()

v = vi.value()
p = vi.policy()

plt.figure()
plt.plot(v, 'b')
plt.show()

plt.figure()
plt.plot(p, 'black')
plt.show()