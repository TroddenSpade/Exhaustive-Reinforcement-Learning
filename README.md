# Exhaustive Reinforcement Learning

## Motivations

## Pseudocode and Algorithms
### Model Taxonomy
* Model-Free Methods

* Model-Based Methods

### Intro To RL Contents
* Tabular Methods
    + Bandit Problem
    + Dynamic Programming
    + Monte Carlo Methods
    + Temporal-Difference Learning
    + n-step Bootstrapping
    + Planning and Learning

* Approximate Solution Methods
    + On-policy Prediction With Approxiamtion
        - Gradient Monte Carlo
        - Semi-Gradient TD(0)
    + On-policy Control With Approxiamtion
        - Semi-Grdient SARSA
        - Semi-Gradient n-step SARSA
    + Off-policy Control  With Approxiamtion
    + Eligibility Traces
    + Policy Gradient Methods
        - REINFORCE
        - one-step Actor-Critic

* Deep Reinforcement Learning Methods
    + Neural Fitted Q-function (NFQ)

* Approximate Solution With Deep Neural Network

## Environments

* Black Jack
    + Monte Carlo Prediction
    + Monte Carlo Exploring Starts

* Random Walk
    + n-step TD Prediction
    + Gradient Monte Carlo State Aggregation
    + Gradient Monte Carlo Tile Coding
    + Semi-Gradient TD(0) State Aggregation

* Mountain Climbing
    + Semi-Gradient SARSA 
    + Semi-Gradient n-step SARSA

* Short Corridor Gridworld
    + REINFORCE [with Baseline]

* Pendulum Swing-Up
    + Actor-Critic using Tile-coding
    + Actor-Critic Countinous Action Space

* CartPole
    + Fully Connected Q-function

## Key Papers

* Deep Reinforcement Learning
    + NFQ :: [Riedmiller, Martin. (2005). Neural fitted Q iteration](https://link.springer.com/chapter/10.1007/11564096_32)
    + DQN :: [V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, & M. Riedmiller. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
    

## Relevant Resourses

* Textbooks
    + [Reinforcement Learning An Introduction, Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book.html)

* Courses
    + Artificial Inteligence
        - [UC Berkeley CS188: Introduction to Artificial Intelligence]()

    + Reinforcement Learning
        - [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/)

    + Deep Reinforcement Learning
        - [UC Berkeley CS285: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)

* Useful Blogs
    + [Lil'Log - Lilian Weng](https://lilianweng.github.io/lil-log/)
    + [Seita's Place - Daniel Seita](https://danieltakeshi.github.io/)
    + [endtoend.ai - Seungjae Ryan Lee](https://endtoend.ai)

## Contribution