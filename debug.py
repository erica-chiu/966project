from env1 import CoopEnv
from env2 import CompEnv
from env3 import MyEnv
from planner import Planner
import math


environment = CoopEnv(1)
p = Planner(environment, CoopEnv(2))
for a in environment.actions:
    print(a)
    print(math.exp(p.competitive_model1.log_probs1[((6 ,8), a)]))
    print(math.exp(p.cooperative_model1.log_probs[((6 ,8), a)]))

