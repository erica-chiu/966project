from env1 import CoopEnv
from env2 import CompEnv
from env3 import MyEnv
from planner import Planner
import math


environment = CompEnv(1)
p = Planner(environment, CompEnv(2))
for a in environment.actions:
    print(a)
    print(math.exp(p.competitive_model.log_probs1[((2 ,3), a)]))
    print(math.exp(p.competitive_model.log_probs2[((3,2), a)]))
    for a2 in environment.actions:
        print((a,a2))
        print(math.exp(p.cooperative_model.log_probs[((2 ,3), a, a2)]))