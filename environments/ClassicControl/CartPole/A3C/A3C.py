import torch
import pickle
import matplotlib.pyplot as plt
import os
import sys

FILE_NAME = "A3C_res.pkl"
FILE_PATH = os.path.join(
    os.path.dirname(__file__), FILE_NAME)
ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, ROOT_PATH)

from code.DRL.AC.A3C.torch import A3C
from code.DRL.optim.torch import SharedAdam
from code.DRL.tools import define_env

if __name__ == "__main__":
    args = {
        "create_env_fn": define_env("CartPole-v1"),
        "actor_kwargs":{
            "hidden_layers": (256,128),
            "activation_fn": torch.nn.functional.relu,
            "optimizer": torch.optim.Adam,
            "learning_rate": 0.0005,
            "grad_max_norm": 1,
            "entropy_loss_weight": 0.001
        },
        "critic_kwargs":{
            "hidden_layers": (256,128),
            "activation_fn": torch.nn.functional.relu,
            "optimizer": torch.optim.Adam,
            "learning_rate": 0.0007,
            "grad_max_norm": float("inf")
        },
        "shared_optimizer": SharedAdam,
        "n_workers": 4,
        "n_step_max": 50,
        "gamma": 0.99
    }

    agent = A3C(**args)
    res = agent.iterate(n_runs=5, max_episodes=600)

    file = open(FILE_PATH, "wb")
    pickle.dump(res, file)
    file.close()

    # file = open(FILE_PATH, 'br')
    # res = pickle.load(file)

    smoothed_times_mean = torch.mean(res['smoothed_times'], axis=0)
    smoothed_times_min = torch.min(res['smoothed_times'], axis=0)[0]
    smoothed_times_max = torch.max(res['smoothed_times'], axis=0)[0]

    smoothed_rewards_mean = torch.mean(res['smoothed_rewards'], axis=0)
    smoothed_rewards_min = torch.min(res['smoothed_rewards'], axis=0)[0]
    smoothed_rewards_max = torch.max(res['smoothed_rewards'], axis=0)[0]

    plt.figure(1)
    plt.plot(smoothed_times_min, 'y', linewidth=1)
    plt.plot(smoothed_times_max, 'y', linewidth=1)
    plt.plot(smoothed_times_mean, 'y', label='NFQ', linewidth=2)
    plt.fill_between(
        torch.arange(len(smoothed_times_mean)), 
        smoothed_times_min, smoothed_times_max, facecolor='y', alpha=0.2)

    plt.figure(2)
    plt.plot([0,1000],[500,500], '--', color='crimson')
    plt.plot(smoothed_rewards_min, 'deepskyblue', linewidth=1)
    plt.plot(smoothed_rewards_max, 'deepskyblue', linewidth=1)
    plt.plot(smoothed_rewards_mean, 'deepskyblue', label='NFQ', linewidth=2)
    plt.fill_between(
        torch.arange(len(smoothed_rewards_mean)), 
        smoothed_rewards_min, smoothed_rewards_max, facecolor='deepskyblue', alpha=0.2)

    plt.show()