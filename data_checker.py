from env1 import CoopEnv
from env2 import CompEnv
from env3 import MyEnv

filename = 'data/env3/need_to_fix_rewards/ruby_lucy3.csv'
env = MyEnv(1)

with open(filename, 'r') as f:
    for i in range(1, 11):
        print("Round "+str(i)+":")
        values = f.readline().split(", ")
        state = env.init_state
        success = True
        for j in range(0, len(values)-1, 4):
            a1 = int(values[j])
            a2 = int(values[j+1])
            loc1 = int(values[j+2])
            loc2 = int(values[j+3])
            new_state = (loc1, loc2)
            if (state, a1, a2, new_state) not in env.transitions:
                print(str((state, a1, a2, new_state)))
                print(j)
                success = False
            if new_state == env.END_STATE:
                if (state, a1, a2, new_state) in env.collab_state:
                    if env.collab_state[(state, a1, a2, new_state)] != int(values[j+4]):
                        print("should be "+str(env.collab_state[(state, a1, a2, new_state)]))
                        success = False
                elif int(values[j+4]) != -100:
                    print("should be -100")
                    success = False
            
            state = new_state
        if success:
            print("OK\n")
