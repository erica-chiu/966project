import random
import math
from scipy.special import log_softmax

class Cooperative():
    def __init__(self, environment, max_iter=10000):
        self.w  = 0.5  #TODO learn this
        self.beta = 4
        self.init_state = environment.init_state
        self.actions = environment.actions
        self.num_actions = len(self.actions)
        self.q = {}  # (state, a1, a2)
        self.discount = 0.9
        self.transition = environment.transitions
        self.states = environment.states
        self.num_states = len(self.states)
        self.rewards = environment.rewards
        self.max_iter = max_iter
        self.log_probs = {}
    

    def train(self):
        for first in range(self.max_iter):
            new_q = {}
            max_diff = 0

            for s in self.states:
                for a1 in self.actions:
                    for a2 in self.actions:
                        new_q[(s,a1, a2)] = 0

                        if (s,a1, a2) in self.q:
                            for s_ in self.states:
                                if (s, a1, a2, s_) in self.transition:
                                    max_val = 0
                                    for a1_ in self.actions:
                                        for a2_ in self.actions:
                                            if self.q[(s_, a1_, a2_)] > max_val:
                                                max_val = self.q[(s_, a1_, a2_)]
                                    
                                    reward = self.w*self.rewards[(s,a1,a2,s_)][0] + (1-self.w)*self.rewards[(s,a1,a2,s_)][1]
                                    new_q[(s,a1, a2)] += self.transition[(s,a1,a2,s_)] * (reward + self.discount* max_val)
                            max_diff = max(max_diff, abs(new_q[(s,a1,a2)] - self.q[(s,a1,a2)]))
            self.q = new_q
            if max_diff < 1e-5 and first != 0:
                break
        
        for s in self.states:
            relative_probs = []
            for a1 in self.actions:
                for a2 in self.actions:
                    relative_probs.append(self.beta*self.q[(s,a1, a2)])
            relative_probs = log_softmax(relative_probs)
            for j, a1 in enumerate(self.actions):
                for k, a2 in enumerate(self.actions):
                    self.log_probs[(s, a1, a2)] = relative_probs[j*self.num_actions + k]
    
    """
    round should be list of (action pair, state)
    """
    def step(self, round):
        state = self.init_state
        prob = 0
        for ((a1, a2), new_state) in round:
            prob += self.log_probs[(state,a1,a2)] + math.log(self.transition[(state,a1,a2,new_state)])
            state = new_state
        return prob
