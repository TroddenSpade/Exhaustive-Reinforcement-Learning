import gym

def define_env(name):
    def create_env():
        return gym.make(name)
    return create_env