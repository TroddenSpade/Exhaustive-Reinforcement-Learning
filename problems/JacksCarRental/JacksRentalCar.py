import numpy as np


def poisson(k, lambd):
    return (lambd ** k * np.exp(-lambd)) / np.math.factorial(k)


class JacksRentalCar:
    def __init__(self):
        self.n_states = 21 * 21
        self.n_actions = 11
        self.actions = range(0, self.n_actions)  # range(-5, 6)
        self.states = range(0, self.n_states)
        self.loc1 = 20
        self.loc2 = 20
        self.dist_l1, self.dist_l2 = self.distribution()  # (3,3) (4,2)

    def distribution(self):
        dist_l1 = np.zeros((21,21))
        dist_l2 = np.zeros((21,21))
        for ret in range(21):
            for rent in range(21):
                dist_l1[ret, rent] = poisson(rent, 3) * poisson(ret, 3)
                dist_l2[ret, rent] = poisson(rent, 4) * poisson(ret, 2)
        return dist_l1, dist_l2

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions