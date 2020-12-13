
from scipy.special import log_softmax
import math
import random

class Punish():
    def __init__(self, environment, cooperative_model, max_iter=10000):
        self.env = environment
        self.other_probs = cooperative_model.log_probs
        self.actions = self.env.actions
        self.num_actions = len(self.actions)
        self.max_iter = max_iter
        self.beta = 4
        self.log_probs = {}
        self.w = 0.5
        self.discount = 0.9

    def train(self):
        q = {}
        for first in range(self.max_iter):
            new_q = {}
            max_diff = 0
            for s in self.env.states:
                for a in self.actions:
                    new_q[(s,a)] = 0
                    if (s,a) in q:                        
                        for s_ in self.env.states:
                            num_actions = 0
                            state_prob = 0
                            total_rewards = 0
                            max_val = -1000
                            for a2 in self.actions:
                                if (s, a, a2, s_) in self.env.transitions:
                                    num_actions += 1
                                    state_prob += self.env.transitions[(s,a,a2,s_)] * math.exp(self.other_probs[((s[1], s[0]),a2)])
                                    rewards = self.env.rewards[(s,a,a2,s_)]
                                    total_rewards += self.w * rewards[0] - (1-self.w) * rewards[1]
                                    if q[(s_,a2)] > max_val:
                                        max_val = q[(s_,a2)]
                            if num_actions == 0:
                                continue
                            total_rewards /= num_actions
                            new_q[(s,a)] += state_prob * (total_rewards + self.discount* max_val)
                        max_diff = max(max_diff, abs(q[(s,a)] - new_q[(s,a)]))
            q = new_q
            if max_diff < 1e-5 and first != 0:
                print(first)
                print("early")
                break

        for s in self.env.states:
            relative_probs = []
            for a in self.actions:
                relative_probs.append(self.beta*q[(s,a)])
            relative_probs = log_softmax(relative_probs)
            for j, a in enumerate(self.actions):
                self.log_probs[(s,a)] = relative_probs[j]
    
    def step(self, round):
        state = self.env.init_state
        prob = 0
        for ((a1, a2), new_state) in round:
            prob += self.log_probs[(state,a1)]
            state = new_state
        return prob


    def next_move(self, state):
        moves = []
        for a in self.actions:
            moves.append(math.exp(self.log_probs[(state, a)]))
        return random.choices(range(self.num_actions), moves)[0]