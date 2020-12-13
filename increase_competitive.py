import math
from scipy.special import softmax, log_softmax
import random

class IncreaseCompetitive():
    def __init__(self, environment, environment2, max_iter=10000):
        self.beta = 4
        self.env1 = environment
        self.env2 = environment2
        self.discount = 0.95
        self.actions = environment.actions
        self.num_actions = len(self.actions)
        self.max_iter = max_iter

        self.total_prob1 = {}  #(state, action)

        for s in self.env1.states:
            for a in self.actions:
                self.total_prob1[(s,a)] = 0
                for s_ in self.env1.states:
                    for a2 in self.actions:
                        if (s, a, a2, s_) in self.env1.transitions:
                            self.total_prob1[(s,a)] += self.env1.transitions[(s,a,a2,s_)]
        
        self.total_prob2 = {}  #(state, action)

        for s in self.env2.states:
            for a in self.actions:
                self.total_prob2[(s,a)] = 0
                for s_ in self.env2.states:
                    for a2 in self.actions:
                        if (s, a, a2, s_) in self.env2.transitions:
                            self.total_prob2[(s,a)] += self.env2.transitions[(s,a,a2,s_)]

        self.log_probs1 = {}
        # self.log_probs2 = {}
    
    def train(self):
        q0 = self.train_q0(self.env1.transitions, self.env1.rewards, self.env1.states, self.total_prob1)
        q1 = self.train_oneside(self.env2.transitions, self.env2.rewards, self.env2.states, q0)
        self.log_probs1 = self.train_oneside(self.env1.transitions, self.env1.rewards, self.env1.states, q1)

    def train_q0(self, transition, rewards, states, total_prob):
        q0 = {}
        for first in range(self.max_iter):
            new_q0 = {}
            max_diff = 0
            for s in states:
                for a in self.actions:
                    new_q0[(s,a)] = 0
                    if (s,a) in q0:
                        for s_ in states:
                            num_actions = 0
                            state_prob = 0
                            total_rewards = 0
                            max_val = -1000
                            for a2 in self.actions:
                                if (s, a, a2, s_) in transition:
                                    num_actions += 1
                                    state_prob += transition[(s,a,a2,s_)]
                                    total_rewards += rewards[(s,a,a2,s_)][1]
                                    if q0[(s_,a2)] > max_val:
                                        max_val = q0[(s_,a2)]
                            if num_actions == 0:
                                continue
                            state_prob = state_prob / total_prob[(s,a)]
                            total_rewards /= num_actions
                            new_q0[(s,a)] += state_prob * (total_rewards + self.discount* max_val)
                        max_diff = max(max_diff, abs(q0[(s,a)] - new_q0[(s,a)]))
            q0 = new_q0
            if max_diff < 1e-5 and first != 0:
                print(first)
                print("early")
                break
            elif first == self.max_iter-1:
                print(max_diff)
        probs = {}
        for s in states:
            relative_probs = []
            for a in self.actions:
                relative_probs.append(self.beta*q0[(s,a)])
            relative_probs = log_softmax(relative_probs)
            for j, a in enumerate(self.actions):
                probs[(s,a)] = relative_probs[j]
        self.test_probs = probs
        return probs


    def train_oneside(self, transition, rewards, states, probs):
        q1 = {}     
        for first in range(self.max_iter):
            new_q1 = {}
            max_diff = 0
            for s in states:
                for a in self.actions:
                    new_q1[(s,a)] = 0
                    if (s,a) in q1:                        
                        for s_ in states:
                            num_actions = 0
                            state_prob = 0
                            total_rewards = 0
                            max_val = -1000
                            for a2 in self.actions:
                                if (s, a, a2, s_) in transition:
                                    num_actions += 1
                                    state_prob += transition[(s,a,a2,s_)] * math.exp(probs[((s[1], s[0]),a2)])
                                    total_rewards += rewards[(s,a,a2,s_)][0]
                                    if q1[(s_,a2)] > max_val:
                                        max_val = q1[(s_,a2)]
                            if num_actions == 0:
                                continue
                            total_rewards /= num_actions
                            new_q1[(s,a)] += state_prob * (total_rewards + self.discount* max_val)
                        max_diff = max(max_diff, abs(q1[(s,a)] - new_q1[(s,a)]))
            q1 = new_q1
            if max_diff < 1e-5 and first != 0:
                print(first)
                print("early")
                break

        final_log_probs = {}
        for s in states:
            relative_probs = []
            for a in self.actions:
                relative_probs.append(self.beta*q1[(s,a)])
            relative_probs = log_softmax(relative_probs)
            for j, a in enumerate(self.actions):
                final_log_probs[(s,a)] = relative_probs[j]
        return final_log_probs

                        
    """
    round should be list of (action pair, state)
    """
    def step(self, round):
        state = self.env1.init_state
        prob = 0
        for ((a1, a2), new_state) in round:
            prob += self.log_probs1[(state,a1)]
            state = new_state
        return prob

    def next_move(self, state):
        moves = []
        for a in self.actions:
            moves.append(math.exp(self.log_probs1[(state, a)]))
        return random.choices(range(self.num_actions), moves)[0]