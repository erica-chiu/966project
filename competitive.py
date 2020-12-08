import math

class Competitive():
    def __init__(self, environment, max_iter=10000):
        self.beta = 4
        self.actions = environment.actions
        self.num_actions = len(self.actions)
        self.q_0 = {}  # (state, action)
        self.q_1 = {}
        self.discount = 0.9
        # self.u = {}
        self.transition = environment.transition
        self.states = environment.states
        self.num_states = len(self.states)
        self.rewards = environment.rewards
        self.max_iter = max_iter
        self.state = environment.init_state
    
    def train(self):
        for _ in range(self.max_iter):
            for s in self.states:
                for a in self.actions:
                    if (s,a) not in self.q_0:
                        self.q_0[(s,a)] = 0
                    else:
                        for s_ in self.states:
                            num_actions = 0
                            total_prob = 0
                            total_rewards = 0
                            max_val = 0
                            for a2 in self.actions:
                                if (s, a, a2, s_) in self.transition:
                                    num_actions += 1
                                    total_prob += self.transition[(s,a,a2,s_)]
                                    total_rewards += self.rewards[(s,a,a2,s_)][1]
                                    if self.q_0[(s_,a2)] > max_val:
                                        max_val = self.q_0[(s_,a2)]
                            if num_actions == 0:
                                continue
                            total_prob /= num_actions
                            total_rewards /= num_actions
                            self.q_0[(s,a)] += total_prob * (total_rewards + self.discount* max_val)
        
            for s in self.states:
                for a in self.actions:
                    if (s,a) not in self.q_1:
                        self.q_1[(s,a)] = 0
                    else:
                        for s_ in self.states:
                            num_actions = 0
                            total_prob = 0
                            total_rewards = 0
                            max_val = 0
                            for a2 in self.actions:
                                if (s, a, a2, s_) in self.transition:
                                    num_actions += 1
                                    total_prob += self.transition[(s,a,a2,s_)] * self.q_0[(s,a2)]
                                    total_rewards += self.rewards[(s,a,a2,s_)][0]
                                    if self.q_1[(s_,a2)] > max_val:
                                        max_val = self.q_1[(s_,a2)]
                            if num_actions == 0:
                                continue
                            total_rewards /= num_actions
                            self.q_1[(s,a)] += total_prob * (total_rewards + self.discount* max_val)
                        
    """
    round should be list of (action pair, state)
    """
    def step(self, round):
        state = self.init_state
        prob = 0
        for ((a1, a2), new_state) in round:
            prob += 4*self.q_1[(state,a1)] + math.log(self.transition[(state,a1,a2,new_state)])
            state = new_state
        return prob
