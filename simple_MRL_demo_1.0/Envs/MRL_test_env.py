from typing import List
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import random
from gymnasium.envs.registration import register
import math

from sb3_contrib import RecurrentPPO
from modulars import EnemyModular,ExitModular,CoinModular

class MRL_test_env(gym.Env):
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
        self.start_pos = [6,6]
        self.agent_pos = self.start_pos.copy()

        self.enemy_index = 2
        self.enemy_modular = EnemyModular('trained_modules/p1/normal_best',self,0)
        self.coin_index  = 3
        self.coin_modular = CoinModular('trained_modules/p2/normal_best',self,1)
        self.medicine_index = 4
        self.medicine_modular = ExitModular('trained_modules/p3/normal_best',self,2)
        self.snare_index = 5

        self.modular_list = [self.enemy_modular,self.coin_modular,self.medicine_modular]
        self.modular_states_list = [0,0,0] 
        self.action_list = [-1,-1,-1]
        self.curr_modular_index = -1
        self.softmax_output =torch.zeros(3)
        self.same_scale_stateValue = []

        self.max_HP =17
        self.curr_HP = self.max_HP
        

        self.enemy_level = enemy_level
        self.price = price

        self.winRate_list = []
        self.goal_list = self.get_goal().copy()

        self.fight_times = 0

        self.stepNum = 0
        self.clear_Num = 0
        self.max_clear = 20
        self.action_log=[]

        

        self.observation_space = spaces.Dict({
            "map":spaces.Box(0,15,(self.width ,self.height),np.int32),
            "state value":spaces.Box(0,100,(3 ,),dtype=np.float16),
            "modular state": spaces.Box(0,3,(3 ,),np.int32),
            "HP":spaces.Discrete(self.max_HP+1),
            "enemyLevel":spaces.Discrete(7),
            "coinPrice":spaces.Discrete(7)
            })
        
        self.action_space = spaces.Discrete(3)


    def reset(self,seed=None, options=None,enemy_level = -1,price = -1):
        self.down_type = "running"
        self.curr_map = self.origin_map.copy()
        self.agent_pos = self.start_pos.copy()
        self.goal_list = self.get_goal().copy()
        
        self.stepNum = 0
        self.modular_states_list = [0,0,0] 
        self.action_list = [-1,-1,-1]
        self.curr_modular_index = -1
        self.softmax_output =torch.zeros(3)
        self.same_scale_stateValue = []

        
        self.winRate_list =self.get_winRate_list(self.enemy_level)
        self.fight_times = 0


        #print("enemy: " + str(self.enemy_level)+",  price: " + str(self.price))
        self.clear_Num = 0
        self.action_log=[]
        self.curr_HP = self.max_HP
        return self._get_obs() , self._get_info()

    def _get_obs(self):
        return {
            "map":np.array(self.curr_map, dtype=np.int32),
            "state value":np.array(self.softmax_output, dtype=np.float16),
            "modular state":np.array(self.modular_states_list, dtype=np.int32),
            "HP":self.curr_HP,
            "enemyLevel":self.enemy_level,
            "coinPrice":self.price

        }

        
    
    def _get_info(self):
        
        return {
            "same_scale_stateValue":self.same_scale_stateValue,
            "state_value_Vector":self.softmax_output,
            "action_log":self.action_log,
            "hp":self.curr_HP
            
        }
    
    def step(self, action):
        total_reward = 0
        if self.curr_modular_index>0:
            if self.curr_modular_index != action :
                if self.modular_states_list[self.curr_modular_index] == 1:
                    total_reward -= 1
                    self.modular_states_list[self.curr_modular_index] = 0

                self.curr_modular_index = action
                self.modular_states_list[self.curr_modular_index] = 1
            else:
                self.modular_states_list[self.curr_modular_index] = 1

        terminated = False
        observation = self._get_obs()
        info = self._get_info()

        for _ in range(5):
            
            reward ,task_over= self.do_action(action)
            total_reward += reward
            

            if  self.stepNum>= 1000:
                terminated = True
                self.log_msg = "time over"
                self.action_log.append(2)
                print("         time over")
                break
            if self.curr_HP <= 0:
                terminated = True
                self.action_log.append(1)
                total_reward -= 500
                print("         deid")
                
                break

            if self.clear_Num >= self.max_clear:
                terminated = True
                total_reward += self.curr_HP*0.2
                self.action_log.append(0)
                print("         clear")
                break


            observation = self._get_obs()
            info = self._get_info()
            if task_over:
                self.modular_states_list[self.curr_modular_index] = 0
                self.reset_map()
                break
        
                    

        
        
        return observation, total_reward, terminated, False, info
    

    def do_action(self,action):
        reward = -0.5
        self.stepNum +=1
        action_over = False
        true_action = self.action_list[action]
        next_x= self.agent_pos[0]
        next_y =self.agent_pos[1]

        if(true_action == 0):#up
            next_y-=1
        elif(true_action == 1):#down
            next_y+=1
        elif(true_action == 2):#right
            next_x+=1
        elif(true_action == 3):#left
            next_x-=1

        
        if(next_x < 0 or next_x >=self.width or next_y < 0 or next_y >=self.height):
            reward -= 5
        else:
            if(self.curr_map[next_x][next_y] == self.enemy_index):
                win = random.randint(1, 100)
                #self.goal_list[0] = [-1,-1]
                self.action_log.append(3)
                
                winRate = self.winRate_list[self.fight_times]
                self.fight_times += 1
                if winRate == 1:
                    reward += 10 + 15*self.enemy_level
                    msg = "fight win"
                else:
                    msg = "fight losen"
                    reward -= 1
                    self.curr_HP-=3
                    
                action_over = self.task_over(msg ,next_x,next_y)
                
            elif (self.curr_map[next_x][next_y] == self.coin_index):
                    #reward += 10* self.price
                    reward += 12* self.price
                    #self.goal_list[1] = [-1,-1]
                    self.action_log.append(4)
                    self.curr_HP-=5
                    action_over = True
                    

                    action_over = self.task_over("get coin ",next_x,next_y)
                    
            elif (self.curr_map[next_x][next_y] == self.medicine_index):
                    #self.goal_list[2] = [-1,-1]
                    self.action_log.append(5)
                    if self.curr_HP == self.max_HP:
                        msg = "recover overflow"
                        reward -=10
                    else:                       
                        hp_befor = self.curr_HP
                        self.curr_HP+=3
                        if self.curr_HP>= self.max_HP :
                            self.curr_HP = self.max_HP
                        reward += (self.curr_HP -  hp_befor)*0.5
                        msg = "recover " +str(hp_befor) +" to "+ str(self.curr_HP)
                    
                    
                    action_over = self.task_over(msg,next_x,next_y)
                    
                    
            elif (self.curr_map[next_x][next_y] == self.snare_index):
                    self.action_log.append(6)
                    reward -= 50
                    #self.goal_list[3] = [-1,-1]
                    action_over = self.task_over("snare ",next_x,next_y)

            else:
                
                self._update_agent_position(next_x,next_y)

        if self.curr_HP <=0:
            self.curr_HP  = 0
        self.action_list = self.update_modular_list()
        
        
        return reward ,action_over

    def update_modular_list(self):
        vl = []
        al = []
        for m in self.modular_list:
            action,state_value = m.get_retrun()
            vl.append(state_value)
            al.append(action)
        self.same_scale_stateValue = torch.tensor(vl)
        if self.dc != None:
            for i in range(len(self.dc)):
                wight = self.dc[i]
                new_sv = self.same_scale_stateValue[i] + wight
                if new_sv > 100:
                    self.same_scale_stateValue[i] = 100
                elif new_sv < 0:
                    self.same_scale_stateValue[i] = 0
                else:
                    self.same_scale_stateValue[i] = new_sv
        self.softmax_output = torch.softmax(self.same_scale_stateValue ,dim= 0)
        
        return al
    
    def task_over(self,msg,next_x,next_y):
        self.log_msg = msg
        print(self.log_msg)
        
        self._update_agent_position(next_x,next_y)
        
        self.clear_Num += 1
        return True


    def _update_agent_position(self, next_x,next_y):
        self.curr_map[self.agent_pos[0], self.agent_pos[1]] = 0
        self.curr_map[next_x, next_y] = 1
        self.agent_pos = [next_x, next_y]

    
    def reset_map(self):
        self.curr_map = self.origin_map.copy()
        self.agent_pos = self.start_pos.copy()
        self.goal_list = self.get_goal().copy()



    def get_distance(self,pos1,pos2):
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance

    def get_goal(self):
        
        res = [[],[],[],[]]
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
            
            
            res[g-2] = [new_x,new_y]

        return res

    def get_random_agentPos(self):
            p =[-1,-1]
            while True:
                x  = random.randint(0, self.width - 1)

                y  = random.randint(0, self.height - 1)

                if self.curr_map[x][y] == 0:
                    p =[x,y]
                    break
            return p

    def get_winRate_list(self, level):
        win_times = int(3 + 7 *(( 5 - level )/2)     )
        winList = [1]*win_times
        loseList = [0]*(self.max_clear-win_times)
        winRate_list = winList+ loseList
        random.shuffle(winRate_list)

        return winRate_list


    
register(
    id='MRL_test_env-v0',
    entry_point='Envs.MRL_test_env:MRL_test_env',
    max_episode_steps=1050,
)
        