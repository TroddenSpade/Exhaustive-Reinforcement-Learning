import torch
import os
import sys
import pickle
import matplotlib.pyplot as plt

PATH = os.path.dirname(__file__)

if __name__ == "__main__":

    list_results = [
        {
            "path":"./DDPG-torch/DDPG_res.pkl",
            "name":"DDPG",
            "color":"g"
        }, 
        {
            "path":"./TD3-torch/TD3_res.pkl",
            "name":"TD3",
            "color":"purple"
        }
    ]
    
    fig_times = plt.figure()
    fig_rewards = plt.figure()
    ax_times = fig_times.add_subplot(1, 1, 1)
    ax_rewards = fig_rewards.add_subplot(1, 1, 1)
    
    for data in list_results:
        file = open(os.path.join(PATH, data["path"]), 'br')
        res = pickle.load(file)

        n_runs, n_episodes = res['rewards'].shape
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

        
        # ax_times.plot(smoothed_times_min, data["color"], linewidth=0.5)
        # ax_times.plot(smoothed_times_max, data["color"], linewidth=0.5)
        ax_times.plot(smoothed_times_mean, data["color"], label=data["name"], linewidth=2)
        ax_times.fill_between(
            torch.arange(len(smoothed_times_mean)), 
            smoothed_times_min, smoothed_times_max, facecolor=data["color"], alpha=0.1)
        
        # ax_rewards.plot(smoothed_rewards_min, data["color"], linewidth=0.5)
        # ax_rewards.plot(smoothed_rewards_max, data["color"], linewidth=0.5)
        ax_rewards.plot(smoothed_rewards_mean, data["color"], label=data["name"], linewidth=2)
        ax_rewards.fill_between(
            torch.arange(len(smoothed_rewards_mean)), 
            smoothed_rewards_min, smoothed_rewards_max, facecolor=data["color"], alpha=0.1)
        ax_rewards.set_yticks(range(-1900, 301, 200))

    ax_times.legend(loc='lower right')
    ax_rewards.legend(loc='lower right')
    plt.show()