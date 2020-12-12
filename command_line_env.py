from env1 import CoopEnv
from env2 import CompEnv
from env3 import MyEnv
import random

envnum = 'env2'
filename = "data/"+ envnum + "/jyen2.csv"
if '1' in envnum:
    env = CoopEnv(1)
elif '2' in envnum:
    env = CompEnv(1)
else:
    env = MyEnv(1)


transition = env.transitions
rewards = env.rewards
moves = {"u": 0, "r": 1, "d": 2, "l": 3, "s": 4}

def print_state(loc1, loc2, height, width, goal1, goal2):
    string_state = ""
    for i in range(height):
        if i != 0:
            for _ in range(width*2+1):
                string_state += "---"
            string_state += "\n"
        for j in range(width):
            idx = j + i * width
            if j != 0:
                string_state += " | "
            
            if loc1 == idx:
                string_state += " A "
            elif loc2 == idx:
                string_state += " B "
            elif idx in goal1:
                string_state += " gA"
            elif idx in goal2:
                string_state += " gB"
            else:
                string_state += "   "
        string_state += "\n"
        
    print(string_state)
    

with open(filename, "w+") as f:
    rounds = []

    for i in range(1, 11):
        new_round = []
        print("\nRound "+str(i) + "\n")
        reward1 = 0
        reward2 = 0
        state = env.init_state
        prev_state = env.init_state
        prev_reward1 = 0
        prev_reward2 = 0
        while True:
            print_state(state[0], state[1], env.height, env.width, env.goal1.keys(), env.goal2.keys())
            var = input("Command: ")
            if 'z' in var:
                state = prev_state
                reward1 = prev_reward1
                reward2 = prev_reward2
                [new_round.pop() for i in range(4)]
                print("\n")
                print("Reward A: "+str(reward1))
                print("Reward B:" + str(reward2))
                continue
            try:
                a,b = var.split(" ")
                a = moves[a]
                b = moves[b]
            except:
                print("Error command. Try again.\n")
                continue
            new_round.append(str(a))
            new_round.append(str(b))
            possible_states = []
            possible_rewards = []
            for s in env.states:
                if (state, a, b, s) in transition:
                    possible_states.append(s)
                    possible_rewards.append(rewards[(state, a, b, s)])
            idx = 0
            if len(possible_states) == 2:
                if random.random() > 0.5:
                    idx = 1
            prev_state = state
            state = possible_states[idx]                
            new_round.append(str(state[0]))
            new_round.append(str(state[1]))
            prev_reward1 = reward1
            prev_reward2 = reward2
            reward1 += possible_rewards[idx][0]
            reward2 += possible_rewards[idx][1]
            print("\n")
            print("Reward A: "+str(reward1))
            print("Reward B: " + str(reward2))
            if state == env.END_STATE:
                if (prev_state, a, b, state) in env.collab_state:
                    new_round.append(str(env.collab_state[(prev_state, a, b, state)]))
                else:
                    new_round.append(str(-100))
                print("End round\n")
                break
        f.write(", ".join(new_round))
        f.write("\n")
        f.flush()
    
    print("Thanks! You're done!")

    

    