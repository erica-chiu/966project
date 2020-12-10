import random
import math

class Cooperative():
    def __init__(self, environment, max_iter=10000):
        self.w  = 0.5  #TODO learn this
        self.beta = 4
        self.actions = environment.actions
        self.num_actions = len(self.actions)
        self.q = {}  # (state, a1, a2)
        self.discount = 0.9
        self.transition = environment.transitions
        self.states = environment.states
        self.num_states = len(self.states)
        self.rewards = environment.rewards
        self.max_iter = max_iter
    

    def train(self):
        for _ in range(self.max_iter):
            for s in self.states:
                for a1 in self.actions:
                    for a2 in self.actions:
                        if (s,a1, a2) not in self.q:
                            self.q[(s,a1, a2)] = 0
                        else:
                            for s_ in self.states:
                                if (s, a1, a2, s_) in self.transition:
                                    max_val = 0
                                    for a1_ in self.actions:
                                        for a2_ in self.actions:
                                            if self.q[(s_, a1_, a2_)] > max_val:
                                                max_val = self.q[(s_, a1_, a2_)]
                                    reward = self.w*self.rewards[(s,a1,a2,s_)][0] + (1-self.w)*self.rewards[(s,a1,a2,s_)][1]
                                    self.q[(s,a1, a2)] += self.transition[(s,a1,a2,s_)] * (reward + self.discount* max_val)
    
    """
    round should be list of (action pair, state)
    """
    def step(self, round):
        state = self.init_state
        prob = 0
        for ((a1, a2), new_state) in round:
            prob += 4*self.q[(state,a1,a2)] + math.log(self.transition[(state,a1,a2,new_state)])
            state = new_state
        return prob
