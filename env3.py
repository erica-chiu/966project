
class MyEnv():
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4

    def move(self, action, loc):
        if action == self.UP:
            if loc < self.width:
                return -1
            return loc-self.width
        elif action == self.RIGHT:
            if loc % self.width == (self.width-1):
                return -1
            return loc + 1
        elif action == self.DOWN:
            if loc > (self.height-1)*(self.width)-1:
                return -1
            return loc + self.width
        elif action == self.LEFT:
            if loc % self.width == 0:
                return -1
            return loc - 1

        
        return loc 

    def __init__(self, agent_idx):
        self.num_actions = 5
        self.actions = list(range(self.num_actions))


        self.width = 4
        self.height = 6
        self.END_STATE = (-1, -1)
        if agent_idx == 1:
            self.end_goals = [21, 22]
            self.other_goals = [1, 2]
            self.init_state = (2, 21)
            self.loc1 = list(range(self.width*self.height))
            [self.loc1.remove(i) for i in self.end_goals]
            self.loc2 = list(range(self.width*self.height))
            [self.loc2.remove(i) for i in self.other_goals]
            
        else:
            self.end_goals = [1, 2]
            self.other_goals = [21, 22]
            self.init_state = (21, 2)
            self.loc1 = list(range(self.width*self.height))
            [self.loc1.remove(i) for i in self.end_goals]
            self.loc2 = list(range(self.width*self.height))
            [self.loc2.remove(i) for i in self.other_goals]
        
        self.states = set()  #(loc1, loc2)
        for loc1 in self.loc1: 
            for loc2 in self.loc2:
                if loc1 != loc2:
                    self.states.add((loc1,loc2))
        self.states.add(self.END_STATE)  # ending

        self.goal1 = {self.end_goals[0]: 10, self.end_goals[1]: 20}
        self.goal2 = {self.other_goals[0]: 10, self.other_goals[1]: 20}
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
                    reward = [0,0]
                    if a1 != self.STAY:
                        reward[0] = -1
                    if a2 != self.STAY:
                        reward[1] = -1
                    if new1 == -1 or new2 == -1:
                        self.transitions[(s, a1, a2, s)] = 1
                        self.rewards[(s, a1, a2, s)] = reward
                        continue

                    if old2 == new1 and old1 == new2:
                        self.transitions[(s, a1, a2, s)] = 1
                        self.rewards[(s, a1, a2, s)] = reward
                        continue

                    if new1 == new2 and (old1 == new1 or old2 == new2):
                        self.transitions[(s, a1, a2, s)] = 1
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
                        
                        if new2 in self.other_goals:
                            s1 = self.END_STATE
                            reward1[1] = self.goal2[new2]
                        if new1 in self.end_goals:
                            s2 = self.END_STATE
                            reward2[0] = self.goal1[new1]
                        
                        self.transitions[(s, a1, a2, s1)] = 0.5
                        self.transitions[(s, a1, a2, s2)] = 0.5
                        self.rewards[(s, a1, a2, s1)] = reward1
                        self.rewards[(s, a1, a2, s2)] = reward2

                        continue

                    s_ = (new1, new2)
                    if new1 in self.end_goals or new2 in self.other_goals:
                        s_ = self.END_STATE
                    if new1 in self.end_goals:
                        reward[0]= self.goal1[new1]
                    if new2 in self.other_goals:
                        reward[1]= self.goal2[new2]
                    if new1 in self.end_goals and new2 in self.other_goals:
                        cooperation_value = 50
                        if self.goal1[new1] == 20 and self.goal2[new2] == 20:
                            cooperation_value = 100
                        elif self.goal1[new1] == 10 and self.goal2[new2] == 10:
                            cooperation_value = 25
                        self.collab_state[(s,a1,a2,s_)] = cooperation_value
                    self.transitions[(s, a1, a2, s_)] = 1
                    self.rewards[(s, a1, a2, s_)] = reward







                    

                    

