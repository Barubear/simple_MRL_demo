
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
def remove_outliers_std(data, threshold=2):
    mean = np.mean(data)
    std = np.std(data)
    
    # 只保留在 [mean - threshold * std, mean + threshold * std] 之间的数据
    filtered_data = [x for x in data if (mean - threshold * std < x < mean + threshold * std)]
    
    return filtered_data


def train(model,env,total_timesteps, save_path,log_path,test_function = None,test_times = None,policy = "MlpLstmPolicy",test_only = False):
   
    if test_function!=None and test_times == None:
        return print("Missing minimum parameter: test_times")
    if not test_only:
   
        msg_pre_tarin =evaluate_policy(model,env,n_eval_episodes=10,deterministic=True)
        
        model.learn(total_timesteps, log_interval=4,progress_bar=True,callback = SaceBaseCallback(save_path,log_path))
        
        
        

    print('model test:')
    if test_function!=None:
        test_function(save_path,env,test_times)
    
    





    

class SaceBaseCallback(BaseCallback):
    def __init__(self, save_path,log_path):
        super().__init__(verbose=0)
        self.best = -float('inf')
        self.save_path = save_path
        self.log_path = log_path
        self.best_step = 0
    def _on_step(self) -> bool:
        
        if self.n_calls%1000 != 0:
            
            return True
        x , y = ts2xy(load_results(self.log_path),'timesteps')
        mean_reward = sum(y[-100:])/len(y[-100:])
        print(self.best_step)
        if mean_reward >self.best:
            self.best = mean_reward
            self.best_step = self.n_calls
            print(self.n_calls,self.best)
            self.model.save(self.save_path)
        
        return True
    
