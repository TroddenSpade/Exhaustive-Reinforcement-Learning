# Exhaustive Reinforcement Learning
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis et augue bibendum, vestibulum neque quis, ultricies mauris. Nunc aliquet velit in nisi rutrum luctus. Etiam nec consequat dui. Phasellus tincidunt, odio non gravida efficitur, enim odio varius dolor, fermentum auctor massa arcu quis ipsum. Sed vel placerat nisi. Nam imperdiet tincidunt facilisis. Curabitur eu leo massa.

## Motivations
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis et augue bibendum, vestibulum neque quis, ultricies mauris. Nunc aliquet velit in nisi rutrum luctus. Etiam nec consequat dui. Phasellus tincidunt, odio non gravida efficitur, enim odio varius dolor, fermentum auctor massa arcu quis ipsum. Sed vel placerat nisi. Nam imperdiet tincidunt facilisis. Curabitur eu leo massa.

## Table of Contents
* [Pseudocode and Algorithms](#pseudocode-and-algorithms)
    - [Textbooks Taxonomy](#textbooks-taxonomy)
* [Implementations and Environments](#implementations-and-environments)
* [Relevant Resourses](#relevant-resourses)
    - [Textbooks](#textbooks)
    - [Courses](#courses)
    - [Useful Blogs](#useful-blogs)
* [Key Papers](#key-papers)
* [Contribution](#contribution)

## Pseudocode and Algorithms

### Textbooks Taxonomy
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
    + Value-based Methods
        - Neural Fitted Q-function (NFQ)
        - DQN
        - DDQN
        - Dueling DDQN
        - PER
    + Policy-based Methods
        - REINFORCE
        - VPG
        - A2C
        - A3C
        - GAE
    + Actor-critic Methods
        - DDPG
        - TD3
        - SAC
        - PPO

## Environments

* Black Jack
    + Monte Carlo Prediction
    + Monte Carlo Exploring Starts

* CartPole
    + Fully Connected Q-function
    + DQN
    + DDQN
    + Dueling DQN

* Cliff Walking
    + SARSA
    + Q-Learning
    + Expected SARSA

* Gambler's Problem
    + Value Iteration

* Grid World
    + Iterative Policy Evaluation

* Jack's Car Rental
    + Policy Iteration

* Lunar Lander
    + REINFORCE using Non-linear Approximation
    + VPG

* Small MDP (Maximization Bias)
    + Q-Learning
    + Double Q-Learning

* Mountain Climbing
    + Semi-Gradient SARSA 
    + Semi-Gradient n-step SARSA

* Multi-Armed Bandit
    + Simple Bandit
    + Gradient Bandit

* Pendulum Swing-Up
    + Actor-Critic using Tile-coding
    + Actor-Critic Countinous Action Space

* Random Walk
    + n-step TD Prediction
    + Gradient Monte Carlo State Aggregation
    + Gradient Monte Carlo Tile Coding
    + Semi-Gradient TD(0) State Aggregation

* Short Corridor Gridworld
    + REINFORCE (Policy Gradient) using Linear Approximation
    + REINFORCE with Baseline

* Windy Grid World
    + SARSA


## Relevant Resourses

### Textbooks
+ [Reinforcement Learning An Introduction, Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book.html)

### Courses
+ Artificial Inteligence
    - [UC Berkeley CS188: Introduction to Artificial Intelligence](https://inst.eecs.berkeley.edu/~cs188/)

+ Reinforcement Learning
    - [UCL Course on Reinforcement Learning](https://www.davidsilver.uk/teaching/)
    - [DeepMind & UCL: Reinforcement Learning Course](https://youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)
    - [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/)

+ Deep Reinforcement Learning
    - [DeepMind x UCL: Deep Learning Lecture Series 2020](https://youtube.com/playlist?list=PLqYmG7hTraZCDxZ44o4p3N5Anz3lLRVZF)
    - [UC Berkeley CS285: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)

### Useful Blogs
+ [Lil'Log - Lilian Weng](https://lilianweng.github.io/lil-log/)
+ [Seita's Place - Daniel Seita](https://danieltakeshi.github.io/)
+ [endtoend.ai - Seungjae Ryan Lee](https://endtoend.ai)


## Key Papers
+ REINFORCE - [Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn. 1992 83 8, 229–256 (1992).](https://link.springer.com/article/10.1007/BF00992696)

* Deep Reinforcement Learning
    * Value-based Methods
        + NFQ - [Riedmiller, M. Neural fitted Q iteration - First experiences with a data efficient neural Reinforcement Learning method. in Lecture Notes in Computer Science vol. 3720 LNAI 317–328 (Springer, Berlin, Heidelberg, 2005).](https://link.springer.com/chapter/10.1007/11564096_32)
        + DQN - [Mnih, V. et al. Playing Atari with Deep Reinforcement Learning. (2013).](https://arxiv.org/abs/1312.5602)
            - [Mnih, V. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).](https://www.nature.com/articles/nature14236)
        + DDQN - [van Hasselt, H., Guez, A. & Silver, D. Deep Reinforcement Learning with Double Q-learning. 30th AAAI Conf. Artif. Intell. AAAI 2016 2094–2100 (2015).](https://arxiv.org/abs/1509.06461)
        + Dueling DQN - [Wang, Z. et al. Dueling Network Architectures for Deep Reinforcement Learning. 33rd Int. Conf. Mach. Learn. ICML 2016 4, 2939–2947 (2015).](https://arxiv.org/abs/1511.06581v3)
    

## Contribution
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis et augue bibendum, vestibulum neque quis, ultricies mauris. Nunc aliquet velit in nisi rutrum luctus. Etiam nec consequat dui. Phasellus tincidunt, odio non gravida efficitur, enim odio varius dolor, fermentum auctor massa arcu quis ipsum. Sed vel placerat nisi. Nam imperdiet tincidunt facilisis. Curabitur eu leo massa.