from env1 import CoopEnv
from env2 import CompEnv
from env3 import MyEnv
from planner import Planner
import math
import random


environment = CoopEnv(1)
p = Planner(environment, CoopEnv(2))
state = environment.init_state
while(True):
    a1 = p.cooperative_model1.next_move(state)
    a2 = p.cooperative_model2.next_move((state[1], state[0]))
    print((a1,a2))
    next_states = []
    next_probs = []
    for s in environment.states:
        if (state, a1, a2, s) in environment.transitions:
            next_states.append(s)
            next_probs.append(environment.transitions[(state, a1, a2, s)])
    next_state = random.choices(next_states, next_probs)[0]
    print(next_state)
    transition = (state, a1, a2, next_state)
    if next_state == environment.END_STATE:
        if transition in environment.collab_state:
            print("collabe")
        else:
            print("not")
        break
    state=next_state


for a in environment.actions:
    print(a)
    # for s1,s2 in [(6,8), (6,5)]:
    #     print(math.exp(p.competitive_model1.log_probs1[((s1 ,s2), a)]))
    #     print(p.competitive_model1.test_probs[((s1,s2), a)])
    for s1,s2 in [(6,8),(3,5), (5,1)]:
        print(math.exp(p.cooperative_model1.log_probs[((s1,s2), a)]))

