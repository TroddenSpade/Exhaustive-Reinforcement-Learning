class Buffer:
    def __init__(self) -> None:
        self.states = []
        self.rewards = []
        self.actions = []

    def store(self, states, actions, rewards):
        self.states.append(states)
        self.rewards.append(rewards)
        self.actions.append(actions)

    def clear(self):
        self.states = []
        self.rewards = []
        self.actions = []

    def get(self):
        return self.states, self.actions, self.rewards
