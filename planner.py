from competitive import Competitive
from cooperative import Cooperative
import random
import math
from scipy.special import log_softmax, softmax

class Planner():

    def __init__(self, environment, environment2, p_coop=1, p_comp=1):
        max_iter = 100
        self.competitive_model = Competitive(environment, environment2, max_iter)
        self.competitive_model.train()
        self.cooperative_model = Cooperative(environment, max_iter)
        self.cooperative_model.train()
        self.environment = environment
        self.p_coop = p_coop
        self.p_comp = p_comp

    """
    0: coop
    1: comp
    """
    def infer(self, rounds):
        if len(rounds) == 0:
            return random.randint(0,1)
        total_coop = 0
        total_comp = 0
        for round in rounds:
            coop = self.cooperative_model.step(round) + math.log(self.p_coop)
            comp = self.competitive_model.step(round) + math.log(self.p_comp)
            probs = log_softmax([coop, comp])
            total_coop += probs[0]
            total_comp += probs[1]
        probs = softmax([total_coop, total_comp])
        if random.random() > probs[0]:
            return 1
        else: 
            return 0




