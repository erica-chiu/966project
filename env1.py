
class CoopEnv():
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4

    def move(self, action, loc):
        if action == self.UP:
            if loc < 3:
                return -1
            return loc-3
        elif action == self.RIGHT:
            if loc % 3 == 2:
                return -1
            return loc + 1
        elif action == self.DOWN:
            if loc > 5:
                return -1
            return loc + 3
        elif action == self.LEFT:
            if loc % 3 == 0:
                return -1
            return loc - 1

        
        return loc 

    def __init__(self, agent_idx):
        self.num_actions = 5
        self.actions = list(range(self.num_actions))


        self.width = 3
        self.height = 3
        self.END_STATE = (-1, -1)
        if agent_idx == 1:
            self.loc1 = list(range(9))
            self.loc1.remove(2)
            self.loc2 = list(range(1,9))
            self.end_goal = 2
            self.other_goal = 0
            self.init_state = (6, 8)

        else:
            self.loc2 = list(range(9))
            self.loc2.remove(2)
            self.loc1 = list(range(1,9))
            self.end_goal = 0
            self.other_goal = 2
            self.init_state = (8, 6)

        self.states = set()  #(loc1, loc2)
        for loc1 in self.loc1: 
            for loc2 in self.loc2:
                if loc1 != loc2:
                    self.states.add((loc1,loc2))
        self.states.add(self.END_STATE)  # ending

        self.goal1 = {self.end_goal: 10}
        self.goal2 = {self.other_goal: 10}
        self.collab_state = {}


        self.rewards = {}  #(s, a1, a2, s')
        # for a2 in self.actions:
        #     for a1 in range(self.num_actions-1):
        #         for s in self.states:
        #             for s_ in self.states:
        #                 self.rewards[(s, a1, a2, s_)] = -1

        # for loc2 in range(9):
        #     for a2 in self.actions:
        #         if agent_idx == 1:
        #             self.rewards[((1, loc2), self.LEFT, a2)] = 10
        #             self.rewards[((5, loc2), self.UP, a2)] = 10
        #         else:
        #             self.rewards[((1, loc2), self.RIGHT, a2)] = 10
        #             self.rewards[((3, loc2), self.UP, a2)] = 10


        self.transitions = {}  #(s, a1, a2, s') s -> s'

        for s in self.states:
            for a1 in self.actions:
                for a2 in self.actions:
                    if s == self.END_STATE:
                        self.transitions[(s, a1, a2, s)] = 1
                        self.rewards[(s, a1, a2, s)] = (0, 0)
                        continue

                    old1 = s[0]
                    old2 = s[1]
                    new1 = self.move(a1, old1)
                    new2 = self.move(a2, old2)
                    if new1 == -1 or new2 == -1:
                        self.transitions[(s, a1, a2, s)] = 1
                        reward = [0,0]
                        if a1 != self.STAY:
                            reward[0] = -1
                        if a2 != self.STAY:
                            reward[1] = -1
                        self.rewards[(s, a1, a2, s)] = reward
                        continue

                    if old2 == new1 and old1 == new2:
                        self.transitions[(s, a1, a2, s)] = 1
                        reward = [0,0]
                        if a1 != self.STAY:
                            reward[0] = -1
                        if a2 != self.STAY:
                            reward[1] = -1
                        self.rewards[(s, a1, a2, s)] = reward
                        continue

                    if new1 == new2 and (old1 == new1 or old2 == new2):
                        self.transitions[(s, a1, a2, s)] = 1
                        reward = [0,0]
                        if a1 != self.STAY:
                            reward[0] = -1
                        if a2 != self.STAY:
                            reward[1] = -1
                        self.rewards[(s, a1, a2, s)] = reward
                        continue
                    
                    if new1 == new2:
                        s1 = (old1, new2)
                        s2 = (new1, old2)
                        reward1 = [0, 0]
                        reward2 = [0, 0]

                        if a1 != self.STAY:
                            reward1[0] = -1
                            reward2[0] = -1
                        if a2 != self.STAY:
                            reward1[1] = -1
                            reward2[1] = -1
                        
                        if new2 == self.other_goal:
                            s1 = self.END_STATE
                            reward1[1] = 10
                        if new1 == self.end_goal:
                            s2 = self.END_STATE
                            reward2[0] = 10
                        
                        self.transitions[(s, a1, a2, s1)] = 0.5
                        self.transitions[(s, a1, a2, s2)] = 0.5
                        self.rewards[(s, a1, a2, s1)] = reward1
                        self.rewards[(s, a1, a2, s2)] = reward2

                        continue

                    s_ = (new1, new2)
                    reward = [0,0]
                    if a1 != self.STAY:
                        reward[0] = -1
                    if a2 != self.STAY:
                        reward[1] = -1
                    if new1 == self.end_goal or new2 == self.other_goal:
                        s_ = self.END_STATE
                    if new1 == self.end_goal:
                        reward[0]= 10
                    if new2 == self.other_goal:
                        reward[1]= 10
                    if new1 == self.end_goal and new2 == self.other_goal:
                        self.collab_state[(s, a1, a2, s_)] = 100
                    self.transitions[(s, a1, a2, s_)] = 1
                    self.rewards[(s, a1, a2, s_)] = reward







                    

                    

