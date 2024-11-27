from typing import List
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random
from gymnasium.envs.registration import register
import math



class p3_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

     
    
    def __init__(self,render_mode = "human",width = 13,height = 13):
        super().__init__()
        self.origin_map = np.transpose(np.array([
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
          
            ], dtype=np.int32))
        self.curr_map = self.origin_map.copy()

        self.width = width
        self.height = height
        self.goal_index = 4

        self.start_pos = [6,6]
        #self.agent_pos = self.start_pos.copy()
        self.HP = 0
        self.hp_index =-1
        self.HP_list = [1,4,8,11,13,14,15,16,17]
        self.goal = self.get_goal().copy()
        self.agent_pos = self.get_random_agentPos().copy()
        
        
        self.curr_dis = self.get_distance(self.agent_pos,self.goal)
        self.min_dis =self.curr_dis
        
        self.stepNum = 0

        self.down_type = "running"
        self.observation_space = spaces.Dict({
            "map":spaces.Box(0,15,(self.width ,self.height),np.int32),
            "agent": spaces.Box(0,15,(2 ,),np.int32),
            "goal": spaces.Box(0,15,(2, ),np.int32),
            'dis': spaces.Discrete(20) ,
            'hp':spaces.Discrete(20)
            })
        
        self.action_space = spaces.Discrete(4)

    def reset(self,seed=None, options=None):

        self.curr_map = self.origin_map.copy()
        #self.agent_pos = self.start_pos.copy()
        self.down_type = "running"
        self.goal = self.get_goal().copy()
        self.agent_pos = self.get_random_agentPos().copy()
        self.curr_dis = self.get_distance(self.agent_pos,self.goal)
        self.min_dis =self.curr_dis
        self.stepNum = 0
        self.hp_index+=1
        self.HP = self.HP_list[self.hp_index% len(self.HP_list)]
        return self._get_obs() , self._get_info()

    def _get_obs(self):
        return {
            "map":np.array(self.curr_map, dtype=np.int32),
            "agent": np.array(self.agent_pos, dtype=np.int32),
            "goal": np.array(self.goal, dtype=np.int32),
            'dis': self.curr_dis ,
            'hp':self.HP
            
        }

        
    
    def _get_info(self):
        
        return {
            "agent": self.agent_pos,
            "down_type":self.down_type
            
        }
    
    def step(self, action):
        reward = -3
        self.curr_dis = self.get_distance(self.agent_pos,self.goal)
        
        self.stepNum +=1
        if  self.curr_dis <= self.min_dis:
            reward +=4
            self.min_dis = self.curr_dis
        
        terminated = False
        next_x= self.agent_pos[0]
        next_y =self.agent_pos[1]

        if(action == 0):#up
            next_y-=1
        elif(action == 1):#down
            next_y+=1
        elif(action == 2):#right
            next_x+=1
        elif(action == 3):#left
            next_x-=1

        
        if(next_x < 0 or next_x >=self.width or next_y < 0 or next_y >=self.height):
            reward -= 5
        else:
            if(self.curr_map[next_x][next_y] == self.goal_index):
                if self.HP<=14:
                    reward += 40
                else:
                    reward += (-10 *  (17-self.HP))

                self._update_agent_position(next_x,next_y)
                terminated = True
                self.arrive = True
                self.down_type = "goal"
                print("goal "+str(self.stepNum))
            else:
                if (self.curr_map[next_x][next_y] == 0):
                    self._update_agent_position(next_x,next_y)
                else:
                    reward -= 30
                    terminated = True
                    self.down_type = "worng point"
                    print("worng point" +str(self.stepNum))

        if  self.stepNum>= 100:
            terminated = True
            self.down_type = "time over"
            print("time over")

            

        
                    

        
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

    

    def _update_agent_position(self, next_x,next_y):
        self.curr_map[self.agent_pos[0], self.agent_pos[1]] = 0
        self.curr_map[next_x, next_y] = 1
        self.agent_pos = (next_x, next_y)

    


    def get_distance(self,pos1,pos2):
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance

    def get_goal(self, point_Num= 4, min_distance = 5):
        
        goal = [-1,-1]
        goals = [2,3,4,5]
        pos = {
            0:[4,4],
            1:[9,4],
            2:[4,9],
            3:[9,9]
        }
        index=[0,1,2,3]
        random.shuffle(goals)
        random.shuffle(index)

        for i in index:
            g = goals[i]
            new_x = pos[i][0]
            new_y = pos[i][1]
            if new_x  > 7:
                new_x +=i
            else:
                new_x -=i
            
            if new_y >7:
                new_y +=i
            else:
                new_y-=i

            self.curr_map[new_x][new_y] = g
            if g == self.goal_index:
                goal =[new_x,new_y]

        return goal

    def get_random_agentPos(self):
            p =[-1,-1]
            while True:
                x  = random.randint(0, self.width - 1)

                y  = random.randint(0, self.height - 1)

                if self.curr_map[x][y] == 0:
                    p =[x,y]
                    break
            return p

        
register(
    id='p3_Env-v0',
    entry_point='Envs.P3_Env:p3_Env',
    max_episode_steps=105,
)
        