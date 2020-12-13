from env1 import CoopEnv
from env2 import CompEnv
from env3 import MyEnv
from planner import Planner
from punish_planner import PunishPlanner
from increase_planner import IncreasePlanner
import os

data = 'data/'
result_dir = 'results/increase/'
env1 = ('env1', CoopEnv(1), CoopEnv(2))
env2 = ('env2', CompEnv(1), CompEnv(2))
env3 = ('env3', MyEnv(1), MyEnv(2))
envs = [env1, env2, env3]
num_rounds = 10

for env_data, environment, environment2 in envs:
    planner = IncreasePlanner(environment, environment2)
    with open(result_dir+env_data+".csv", "w+") as r:
        for filename in os.listdir(data+env_data):
            print(filename)
            rounds1 = []
            rounds2 = []
            results = []
            predictions = []
            with open(data+env_data+"/"+filename, 'r') as f:
                for _ in range(num_rounds):
                    round_str = f.readline().split(", ")
                    result = int(round_str[-1])
                    round1 = []
                    round2 = []
                    for i in range(0, len(round_str)-1, 4):
                        a1 = int(round_str[i])
                        a2 = int(round_str[i+1])
                        loc1 = int(round_str[i+2])
                        loc2 = int(round_str[i+3])
                        round1.append(((a1, a2), (loc1, loc2)))
                        round2.append(((a2, a1), (loc2, loc1)))
                    rounds1.append(round1)
                    rounds2.append(round2)
                    results.append(str(result))
                    prediction = planner.infer(rounds1, rounds2)
                    predictions.append(str(prediction))
            r.write(", ".join(results)+"\n")
            r.write(", ".join(predictions)+"\n")
            r.flush()
        


            
            
            
            
            
            

