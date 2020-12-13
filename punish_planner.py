from punish import Punish
from competitive import Competitive
from cooperative import Cooperative
import random
import math
from scipy.special import log_softmax, softmax

class PunishPlanner():

    def __init__(self, environment, environment2, p_coop=1, p_comp=1, p_punish=1):
        max_iter = 1000
        self.competitive_model1 = Competitive(environment, environment2, max_iter)
        self.competitive_model1.train()
        self.competitive_model2 = Competitive(environment2, environment, max_iter)
        self.competitive_model2.train()
        self.cooperative_model1 = Cooperative(environment, max_iter)
        self.cooperative_model1.train()
        self.cooperative_model2 = Cooperative(environment2, max_iter)
        self.cooperative_model2.train()
        self.punish1 = Punish(environment, self.cooperative_model2, max_iter)
        self.punish1.train()
        self.punish2 = Punish(environment2, self.cooperative_model1, max_iter)
        self.punish2.train()
        
        self.models1 = [self.cooperative_model1, self.competitive_model1, self.punish1]
        self.models2 = [self.cooperative_model2, self.competitive_model2, self.punish2]
        self.environment1 = environment
        self.environment2 = environment2
        self.p_coop = p_coop
        self.p_comp = p_comp
        self.p_punish = p_punish

    """
    0: coop
    1: comp
    """
    def infer(self, rounds1, rounds2):
        model2 = self.models2[self.intention(rounds1, self.cooperative_model1, self.competitive_model1, self.punish1)]
        model1 = self.models1[self.intention(rounds2, self.cooperative_model2, self.competitive_model2, self.punish2)]

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

    
    def intention(self, rounds, coop_model, comp_model, punish_model):
        if len(rounds) == 0:
            return random.randint(0,2)
        total_coop = 0
        total_comp = 0
        total_punish = 0
        for round in rounds:
            coop = coop_model.step(round) + math.log(self.p_coop)
            comp = comp_model.step(round) + math.log(self.p_comp)
            punish = punish_model.step(round) + math.log(self.p_punish)
            probs = log_softmax([coop, comp, punish])
            total_coop += probs[0]
            total_comp += probs[1]
            total_punish += probs[2]
        probs = softmax([total_coop, total_comp, total_punish])
        return random.choices(range(3), probs)[0]

