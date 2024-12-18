import numpy as np
import torch
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

def get_state_value(model,state,obs,device='cuda',policy = "MlpLstmPolicy"):

    # 将obs字典中的每个值转换为PyTorch张量，并放入新的字典中
    obs_tensor = None
    
    if policy == "MultiInputLstmPolicy":
        obs_tensor = {key: torch.as_tensor(obs, device=device) for (key, obs) in obs.items()}
    elif policy == "MlpLstmPolicy":
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    _states_tensor = torch.tensor(state,dtype=torch.float32).to(device)
    episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
    state_value = model.policy.predict_values(obs_tensor,_states_tensor ,episode_starts)
    return state_value.to('cpu')    

def write_log(path,data,tile_list=None,write_type ='w'):
        with open(path, write_type,newline='') as f:
                writer = csv.writer(f)
                if tile_list != None:
                    writer.writerow(tile_list)
                for msg in data:
                    if isinstance(msg, torch.Tensor):
                        msg = msg.tolist()
                    if isinstance(msg, (float, int)):  # 如果是数字类型
                        msg = [msg]
                    writer.writerow(msg)

def get_data_scale(data):
    
    state_value_list = np.array(data)
    print(state_value_list.max())
    print(state_value_list.min())
    print(state_value_list.mean())
    
def remove_outliers_std(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    
    # 只保留在 [mean - threshold * std, mean + threshold * std] 之间的数据
    filtered_data = [x for x in data if (mean - threshold * std < x < mean + threshold * std)]
    
    return filtered_data

def draw_kdeplot(data,title = "Data Distribution - KDE Plot", xlabel = "Value",save_path = None):
    sns.kdeplot(np.array(data).flatten(), fill=True )
    plt.xlabel(xlabel)
    plt.title(title)
    if save_path!= None:
        plt.savefig(title+'.png', format='png', dpi=300)
    else:
        plt.show()
    plt.close()

def draw_boxplot(data,title = "Data Distribution - KDE Plot", ylabel= "Value", xlabel = "Data",save_path = None):
    plt.boxplot(np.array(data).flatten())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_path!= None:
        plt.savefig(title+'.png', format='png', dpi=300)
    else:
        plt.show()
    plt.close()











def ctrl_test( module_path,env,test_times,save_path ="simple_MRL_demo_2.0/test_log\ctrl_log02" ):
    
    print('model test:')

    state_value_list_enemy = []
    state_value_list_coin = []
    state_value_list_exit = []
    state_value_Vector=[]
    action_list=[]
    clear_step =[]
    msg = ''
    model = RecurrentPPO.load( module_path)
    model.set_env(env)

    for i in range(test_times):
        if i %100 == 0:
            print(i)

        step = 0
        obs= env.reset()

        while True:
                
                action, _states = model.predict(obs)
                obs, rewards, dones, info  = env.step(action)
                
                if step != 0:
                    #取出各模块的state value
                    
                    #print(info[0]["same_scale_stateValue"])
                    state_value_list_enemy.append(info[0]["same_scale_stateValue"][0])
                    state_value_list_coin.append(info[0]["same_scale_stateValue"][1])
                    state_value_list_exit.append(info[0]["same_scale_stateValue"][2])
                    #取出经sofmax处理过的state value 向量
                    state_value_Vector.append(info[0]["state_value_Vector"]) 



                step +=1
                if step >= 1000:
                    
                    msg = info[0]["action_log"]
                    hp = info[0]["hp"]
                    action_list.append([msg,hp])
                    break
                elif dones :
                    
                    
                    msg = info[0]["action_log"]
                    hp = info[0]["hp"]
                    action_list.append([msg,hp])
                    break
    
    write_log(save_path+"/state_value_Vector.csv",state_value_Vector)
    write_log(save_path+"/action_log.csv",action_list)
    write_log(save_path+"/enemy_state_value.csv",state_value_list_enemy)
    write_log(save_path+"/coin_state_value.csv",state_value_list_coin)
    write_log(save_path+"/recover_state_value.csv",state_value_list_exit)
    get_data_scale(state_value_list_enemy)
    print("coin")
    get_data_scale(state_value_list_coin)
    print("exit")
    get_data_scale(state_value_list_exit)
    
    draw_boxplot(state_value_list_enemy,title="box_enemy",save_path=save_path)
    draw_boxplot(state_value_list_coin,title="box_coin",save_path=save_path)
    draw_boxplot(state_value_list_exit,title="box_recover",save_path=save_path)

    draw_kdeplot(state_value_list_enemy,title="kde_enemy",save_path=save_path)
    draw_kdeplot(state_value_list_coin,title="kde_coin",save_path=save_path)
    draw_kdeplot(state_value_list_exit,title="kde_recover",save_path=save_path)

    print("over")


def modular_test(modelPath,env,test_times,savePath = None):
    state_value_list = []
    clear_step =[]
    model = RecurrentPPO.load(modelPath)
    model.set_env(env)

    clear_times = 0
    for i in range(test_times):
        step = 0
        obs= env.reset()
        while True:

                action, _states = model.predict(obs)
                obs, rewards, dones, info  = env.step(action)
                state_value_list.append(get_state_value(model,_states,obs,policy = "MultiInputLstmPolicy"))
                step +=1
                if step >= 100:
                    clear_step.append(step)
                    break

                if dones :
                    
                    clear_times += 1
                    clear_step.append(step)
                    
                    break
                
    print("state value")
    #state_value_list = np.array(state_value_list)
    state_value_list = [sv.detach().cpu().numpy() for sv in state_value_list]

    draw_kdeplot(state_value_list)
    draw_boxplot(state_value_list)

    #state_value_list = remove_outliers_std(state_value_list)
    state_value_list =np.array(state_value_list)
    get_data_scale(state_value_list)

    print("step")
    clear_step = np.array(clear_step)
    print(clear_step.mean())
    print(clear_times/test_times)


