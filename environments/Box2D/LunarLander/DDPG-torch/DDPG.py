import torch
import pickle
import matplotlib.pyplot as plt
import os
import sys

PATH = os.path.dirname(__file__)
FILE_PATH = os.path.join(PATH ,"DDPG_res.pkl")
GIF_PATH = os.path.join(PATH , "gifs")
ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, ROOT_PATH)

from code.DRL.DPG.DDPG.torch import DDPG
from code.DRL.tools import DefineEnv

if __name__ == "__main__":
    n_runs = 5
    n_episodes = 1000
    args = {
        "create_env_fn": DefineEnv("LunarLanderContinuous-v2"),
        "actor_kwargs":{
            "hidden_layers":(400,300),
            "activation_fn":torch.nn.functional.relu,
            "out_activation_fn":torch.tanh,
            "optimizer":torch.optim.Adam, 
            "learning_rate":1e-4,
            "grad_max_norm":float("inf"),
        },
        "critic_kwargs":{
            "states_hidden_layers":(400,),
            "shared_hidden_layers":(300,),
            "activation_fn":torch.nn.functional.relu,
            "optimizer":torch.optim.Adam, 
            "learning_rate":1e-3,
            "grad_max_norm":float("inf")
        },
        "gamma":0.99,
        "tau":0.001,
        "batch_size":64
    }

    agent = DDPG(**args)
    res = agent.iterate(n_runs, n_episodes)
    agent.create_gifs(GIF_PATH)

    file = open(FILE_PATH, "wb")
    pickle.dump(res, file)
    file.close()

    # file = open(FILE_PATH,'br')
    # res = pickle.load(file)

    avg_times= torch.zeros((n_runs, n_episodes))
    avg_rewards = torch.zeros((n_runs, n_episodes))

    for r in range(n_runs):
        avg_times[r] = torch.tensor([
            torch.mean(res["times"][r][max(i-9,0):i+1]) for i in range(n_episodes)])
        avg_rewards[r] = torch.tensor([
            torch.mean(res["rewards"][r][max(i-99,0):i+1]) for i in range(n_episodes)])

    smoothed_times_mean = torch.mean(avg_times, axis=0)
    smoothed_times_min = torch.min(avg_times, axis=0)[0]
    smoothed_times_max = torch.max(avg_times, axis=0)[0]

    smoothed_rewards_mean = torch.mean(avg_rewards, axis=0)
    smoothed_rewards_min = torch.min(avg_rewards, axis=0)[0]
    smoothed_rewards_max = torch.max(avg_rewards, axis=0)[0]

    plt.figure(1)
    plt.plot(smoothed_times_min, 'y', linewidth=1)
    plt.plot(smoothed_times_max, 'y', linewidth=1)
    plt.plot(smoothed_times_mean, 'y', label='DDPG', linewidth=2)
    plt.fill_between(
        torch.arange(len(smoothed_times_mean)), 
        smoothed_times_min, smoothed_times_max, facecolor='y', alpha=0.2)

    plt.figure(2)
    plt.plot([0,1000],[200,200], '--', color='crimson')
    plt.plot(smoothed_rewards_min, 'deepskyblue', linewidth=1)
    plt.plot(smoothed_rewards_max, 'deepskyblue', linewidth=1)
    plt.plot(smoothed_rewards_mean, 'deepskyblue', label='DDPG', linewidth=2)
    plt.fill_between(
        torch.arange(len(smoothed_rewards_mean)), 
        smoothed_rewards_min, smoothed_rewards_max, facecolor='deepskyblue', alpha=0.2)

    plt.show()