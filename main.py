from env1 import CoopEnv
from env2 import CompEnv
from env3 import MyEnv
from planner import Planner

environment = CoopEnv(1)
planner = Planner(environment)
data = 'data/'
env1 = ('env1/', CoopEnv(1))
env2 = ('env2/', CompEnv(1))
env3 = ('env3/', MyEnv(1))
