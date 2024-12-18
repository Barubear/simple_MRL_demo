from sb3_contrib import RecurrentPPO
import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np
import torch
import math

class Modulars(ABC):
    def __init__(self,path,env,index,max_state_value,min_state_value) :
        custom_objects = {
        "clip_range": 0.0,
        "lr_schedule": 0.0,
        }
        self.module = RecurrentPPO.load(path,custom_objects=custom_objects)
        self.device = torch.device('cuda' )
        self.main_env =env
        self.max_state_value=max_state_value
        self.min_state_value=min_state_value
        self.index = index
        self.data_scale = [0,100]

    @abstractmethod
    def get_obs(self):
        pass

    def get_distance(self,pos1,pos2):
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance

    def get_retrun(self):
        obs = self.get_obs()
        action, _states = self.module.predict(obs)
        _states = np.array(_states)
        obs_tensor_dict = self.module.policy.obs_to_tensor(obs)[0]
        

        _states_tensor = torch.tensor(_states,dtype=torch.float32).to(self.device)
        episode_starts = torch.tensor([True] ,dtype=torch.float32).to(self.device)

        #print("debug")
        state_value = self.module.policy.predict_values(obs_tensor_dict,_states_tensor ,episode_starts).item()
        
        
        value = self.scale_to_range(state_value,self.min_state_value,self.max_state_value)
        
        return action,value

    def scale_to_range(self,data,old_min, old_max , new_min=0, new_max=100):

        if data >= old_max:
            data = old_max
        elif data <= old_min:
            data = old_min
       
        scaled_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

        return scaled_data


class EnemyModular(Modulars):
    def __init__(self,path,env,index):
        super().__init__(path,env,index,60,-90)
        self.action_state = 0
        #self.max_state_value = 30
        #self.min_state_value = 0
        
        
        
        
    def get_obs(self):
        goal = self.main_env.goal_list[self.index]
        agent = self.main_env.agent_pos
        dis = self.get_distance(goal,agent)
        return {
            "map":np.array(self.main_env.curr_map, dtype=int),
            "agent": np.array(agent, dtype=int),
            "goal": np.array(goal, dtype=np.int32),
            'dis': dis ,
            'level':self.main_env.enemy_level,
            'hp':self.main_env.curr_HP,
        }
        
    

class CoinModular(Modulars):
    def __init__(self,path,env,index) :
        super().__init__(path,env,index,45,-50)
        #self.max_state_value = 50
        #self.min_state_value = 25
        
        
        
    

    def get_obs(self):
        goal = self.main_env.goal_list[self.index]
        agent = self.main_env.agent_pos
        dis = self.get_distance(goal,agent)
        return {
            "map":np.array(self.main_env.curr_map, dtype=int),
            "agent": np.array(agent, dtype=int),
            "goal": np.array(goal, dtype=np.int32),
            'dis': dis ,
            'price':self.main_env.price,
            'hp':self.main_env.curr_HP,
        }
    
    
    

class ExitModular(Modulars):
    def __init__(self,path,env,index) :
        super().__init__(path,env,index,50,-30)
        #self.max_state_value = 50
        #self.min_state_value = -35
        
        
    

    def get_obs(self):
        goal = self.main_env.goal_list[self.index]
        agent = self.main_env.agent_pos
        dis = self.get_distance(goal,agent)
        #print(self.main_env.curr_HP)
        return {
            "map":np.array(self.main_env.curr_map, dtype=int),
            "agent": np.array(agent, dtype=int),
            "goal": np.array(goal, dtype=np.int32),
            'dis': dis ,
            'hp':self.main_env.curr_HP
        }
    
    
    
class BattleModular(Modulars):
    def __init__(self,path,env,index) :
        super().__init__(path,env,index,104,-104)
        #self.max_state_value = 50
        #self.min_state_value = -35
        
        
    

    def get_obs(self):
        goal = self.main_env.goal_list[self.index]
        agent = self.main_env.agent_pos
        dis = self.get_distance(goal,agent)
        #print(self.main_env.curr_HP)
        return {
            "map":np.array(self.main_env.curr_map, dtype=int),
            "agent": np.array(agent, dtype=int),
            "goal": np.array(goal, dtype=np.int32),
            'dis': dis ,
            'ennemylevel':self.main_env.enemy_level,
            'agentlevel':self.main_env.agent_level,
            'hp':self.main_env.curr_HP
        }

    """
    def get_retrun(self):
        obs = self.get_obs()
        action, _states = self.module.predict(obs)
        _states = np.array(_states)
        obs_tensor_dict = self.module.policy.obs_to_tensor(obs)[0]
        

        _states_tensor = torch.tensor(_states,dtype=torch.float32).to(self.device)
        episode_starts = torch.tensor([True] ,dtype=torch.float32).to(self.device)

        #print("debug")
        state_value = self.module.policy.predict_values(obs_tensor_dict,_states_tensor ,episode_starts).item()
        
        subvalue = self.scale_to_range(state_value,self.min_state_value,self.max_state_value,-90,60)
        value = self.scale_to_range(subvalue,self.min_state_value,self.max_state_value)
        
        return action,value
    """
    