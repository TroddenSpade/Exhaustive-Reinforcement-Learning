from algorithms.DP.IterativePolicyEvaluation import IterativePolicyEvaluation
from problems.GridWolrdIPE.GridWorld import GridWorld

ipe = IterativePolicyEvaluation(GridWorld, theta=0.00001, gamma=1)
ipe.evaluate()
grid = ipe.get_V().reshape(4, 4)
print(grid)
