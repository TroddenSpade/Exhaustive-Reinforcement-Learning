# Exhaustive Reinforcement Learning
This repository aims to exhaustively implement various Deep Reinforcement Learning concepts covering most of the well-known resources from textbooks to lectures. For each notion, concise notes are provided to explain, and associated algorithms are implemented in addition to their environments and peripheral modules. At the end of this readme file, Reinforcement Learning's key papers and worthwhile resources are cited.

## Motivations


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
    + Value-Based Methods
        - Neural Fitted Q-function (NFQ)
        - DQN
        - DDQN
        - Dueling DDQN
        - PER
        - C51
        - QR-DQN
        - HER
    + Policy-Based Methods
        - REINFORCE
        - VPG
        - PPO
        - TRPO
    + Stochastic Actor-Critic Methods
        - A2C
        - A3C
        - GAE
        - ACKTR
    + Deterministic Actor-Critic Methods
        - Deep Deterministic Policy Gradient (DDPG)
        - TD3
        - SAC

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
+ [Reinforcement Learning An Introduction. Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book.html)
+ [Algorithms for Reinforcement Learning. Csaba Szepesvari](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
+ [Foundations of Deep Reinforcement Learning: Theory and Practice in Python. Laura Graesser and Wah Loon Keng](https://www.oreilly.com/library/view/foundations-of-deep/9780135172490/)
+ [Grokking Deep Reinforcement Learning. Miguel Morales](https://www.manning.com/books/grokking-deep-reinforcement-learning?query=reinforcement%20learning)
+ [Deep Reinforcement Learning Hands-On: Apply modern RL methods to practical problems of chatbots, robotics, discrete optimization, web automation, and more, 2nd Edition. Maxim Lapan](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
+ [Deep Reinforcement Learning with Python, Second Edition. Sudharsan Ravichandiran](https://www.packtpub.com/product/deep-reinforcement-learning-with-python-second-edition/9781839210686)
+ [Deep Reinforcement Learning in Action, Brandon Brown and Alexander Zai](https://www.manning.com/books/deep-reinforcement-learning-in-action)
+ [Deep Reinforcement Learning Fundamentals, Research and Applications. Hao Dong, Zihan Ding, and Shanghang Zhang](https://www.springer.com/gp/book/9789811540943#)

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
    - [UC Berkeley CS287: Advanced Robotics](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/)

### Useful Blogs
+ [Lil'Log - Lilian Weng](https://lilianweng.github.io/lil-log/)
+ [Seita's Place - Daniel Seita](https://danieltakeshi.github.io/)
+ [endtoend.ai - Seungjae Ryan Lee](https://endtoend.ai)

### Articles
+ [Better Exploration with Parameter Noise](https://openai.com/blog/better-exploration-with-parameter-noise/)


## Key Papers
+ Actor-Critic - []()
+ REINFORCE - [Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn. 1992 83 8, 229–256 (1992).](https://link.springer.com/article/10.1007/BF00992696)

* Deep Reinforcement Learning
    * Value-based Methods
        + NFQ - [Riedmiller, M. Neural fitted Q iteration - First experiences with a data efficient neural Reinforcement Learning method. in Lecture Notes in Computer Science vol. 3720 LNAI 317–328 (Springer, Berlin, Heidelberg, 2005).](https://link.springer.com/chapter/10.1007/11564096_32)
        + DQN - [Mnih, V. et al. Playing Atari with Deep Reinforcement Learning. (2013).](https://arxiv.org/abs/1312.5602)
            - [Mnih, V. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).](https://www.nature.com/articles/nature14236)
        + DDQN - [van Hasselt, H., Guez, A. & Silver, D. Deep Reinforcement Learning with Double Q-learning. 30th AAAI Conf. Artif. Intell. AAAI 2016 2094–2100 (2015).](https://arxiv.org/abs/1509.06461)
        + Dueling DQN - [Wang, Z. et al. Dueling Network Architectures for Deep Reinforcement Learning. 33rd Int. Conf. Mach. Learn. ICML 2016 4, 2939–2947 (2015).](https://arxiv.org/abs/1511.06581v3)
        + PER - [Schaul, T., Quan, J., Antonoglou, I. & Silver, D. Prioritized Experience Replay. 4th Int. Conf. Learn. Represent. ICLR 2016 - Conf. Track Proc. (2015).](https://arxiv.org/abs/1511.05952)
        + Rainbow - [Hessel, M. et al. Rainbow: Combining Improvements in Deep Reinforcement Learning. 32nd AAAI Conf. Artif. Intell. AAAI 2018 3215–3222 (2017).](https://arxiv.org/abs/1710.02298)

    * Policy-based Methods        

    * Actor-Critic Methods
        + AC
        + A3C/A2C - [Mnih, V. et al. Asynchronous Methods for Deep Reinforcement Learning. 33rd Int. Conf. Mach. Learn. ICML 2016 4, 2850–2869 (2016).](https://arxiv.org/abs/1602.01783)
        + GAE - [Schulman, J., Moritz, P., Levine, S., Jordan, M. I. & Abbeel, P. High-dimensional continuous control using generalized advantage estimation. in 4th International Conference on Learning Representations, ICLR 2016 - Conference Track Proceedings (International Conference on Learning Representations, ICLR, 2016).](https://arxiv.org/abs/1506.02438)
        + PPO
        + TRPO
        
    * Deterministic Actor-Critic Methods
        + DPG - [Silver, D. et al. Deterministic Policy Gradient Algorithms. 387–395 (2014).](http://proceedings.mlr.press/v32/silver14.html)
        + DDPG - [Lillicrap, T. P. et al. Continuous control with deep reinforcement learning. 4th Int. Conf. Learn. Represent. ICLR 2016 - Conf. Track Proc. (2015).](https://arxiv.org/abs/1509.02971v6)
        + TD3 - [Fujimoto, S., van Hoof, H. & Meger, D. Addressing Function Approximation Error in Actor-Critic Methods. 35th Int. Conf. Mach. Learn. ICML 2018 4, 2587–2601 (2018).](https://arxiv.org/abs/1801.01290v2)
        + SAC - [Haarnoja, T., Zhou, A., Abbeel, P. & Levine, S. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. 35th Int. Conf. Mach. Learn. ICML 2018 5, 2976–2989 (2018).](https://arxiv.org/abs/1801.01290v2)
    

## Contribution
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis et augue bibendum, vestibulum neque quis, ultricies mauris. Nunc aliquet velit in nisi rutrum luctus. Etiam nec consequat dui. Phasellus tincidunt, odio non gravida efficitur, enim odio varius dolor, fermentum auctor massa arcu quis ipsum. Sed vel placerat nisi. Nam imperdiet tincidunt facilisis. Curabitur eu leo massa.
