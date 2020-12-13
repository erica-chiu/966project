from competitive import Competitive
from cooperative import Cooperative
import random
import math
from scipy.special import log_softmax, softmax

class Planner():

    def __init__(self, environment, environment2, p_coop=1, p_comp=1):
        max_iter = 100
        self.competitive_model1 = Competitive(environment, max_iter)
        self.competitive_model1.train()
        self.competitive_model2 = Competitive(environment2, max_iter)
        self.competitive_model2.train()
        self.cooperative_model1 = Cooperative(environment, max_iter)
        self.cooperative_model1.train()
        self.cooperative_model2 = Cooperative(environment2, max_iter)
        self.cooperative_model2.train()
        self.models1 = [self.cooperative_model1, self.competitive_model1]
        self.models2 = [self.cooperative_model2, self.competitive_model2]
        self.environment1 = environment
        self.environment2 = environment2
        self.p_coop = p_coop
        self.p_comp = p_comp

    """
    0: coop
    1: comp
    """
    def infer(self, rounds1, rounds2):
        model2 = self.models2[self.intention(rounds1, self.cooperative_model1, self.competitive_model1)]
        model1 = self.models1[self.intention(rounds2, self.cooperative_model2, self.competitive_model2)]

        state = self.environment1.init_state
        while(True):
            a1 = model1.next_move(state)
            a2 = model2.next_move((state[1], state[0]))
            next_states = []
            next_probs = []
            for s in self.environment1.states:
                if (state, a1, a2, s) in self.environment1.transitions:
                    next_states.append(s)
                    next_probs.append(self.environment1.transitions[(state, a1, a2, s)])
            next_state = random.choices(next_states, next_probs)[0]
            transition = (state, a1, a2, next_state)
            if next_state == self.environment1.END_STATE:
                if transition in self.environment1.collab_state:
                    return self.environment1.collab_state[transition]
                else:
                    return -100
            state = next_state
            print((a1,a2,state))

    
    def intention(self, rounds, coop_model, comp_model):
        if len(rounds) == 0:
            return random.randint(0,1)
        total_coop = 0
        total_comp = 0
        for round in rounds:
            coop = coop_model.step(round) + math.log(self.p_coop)
            comp = comp_model.step(round) + math.log(self.p_comp)
            probs = log_softmax([coop, comp])
            total_coop += probs[0]
            total_comp += probs[1]
        probs = softmax([total_coop, total_comp])
        if random.random() > probs[0]:
            return 0
        else: 
            return 1




