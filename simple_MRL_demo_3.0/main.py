
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

import train
import procces_statistic_testlog

from Envs.P1_Env import p1_Env
from Envs.P2_Env import p2_Env
from Envs.P3_Env import p3_Env
from Envs.Dctest_env import Dctest_env
from Envs.Modular_test_env import Modular_test_env
from Envs.MRL_env import MRL_env
from Envs.MRL_test_env import MRL_test_env
import test_tool

import os


def Moduletrain(save_path,log_path,env,times = 2000000,testonly =False,re_tarin_model =None,policy = "MultiInputLstmPolicy",test_function=test_tool.modular_test,test_time = 1000):
    model = None
    if re_tarin_model == None:
        model = RecurrentPPO(
        policy ,#
        env,
        learning_rate=1e-4,  # 学习率
        gamma=0.995,  # 折扣因子
        gae_lambda=0.95,  # GAE λ
        clip_range=0.2,  # 剪辑范围
        ent_coef=0.1,  # 熵系数
        batch_size=512,  # 批大小
        n_steps=512,  # 步数
        n_epochs=8,  # 训练次数
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1),  # LSTM 设置
        verbose=1,
        )
    else :
        model = re_tarin_model
        model.set_env(env)
    
    train.train(model,env,times,save_path,log_path,test_function=test_function,test_times=test_time,policy=policy,test_only=testonly)


def train_navi_p1():
    save_path = 'trained_modules/p1/normal_best02'
    log_path = 'logs/p1_log'
    env = make_vec_env("p1_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,2000000,testonly=True)
    """
state value
52.019547
-83.966324
11.514604

step
21.624
0.999
    """




def train_navi_p2():
    save_path = 'trained_modules/p2/normal_best03'
    log_path = 'logs/p2_log'
    env = make_vec_env("p2_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,1000000)

    """
    04
state value
47.648846
-48.851852
15.714743
step
30.915
0.941
    """



def train_navi_p3():
    save_path = 'trained_modules/p3/normal_best03'
    log_path = 'logs/p3_log'
    env = make_vec_env("p3_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,1000000)

"""
02
state value
46.291714
-26.392736
27.115408
step
0.955
1.0


03
state value
38.848442
-29.62391
19.911083
step
13.507
1.0
"""


def train_ctrl():
    save_path = 'trained_modules/ctrl/normal_bestB2'
    log_path = 'logs/ctrl_log'
    env = make_vec_env("MRL_env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,2000000,test_function=test_tool.ctrl_test, testonly= True)

    """
    92.68228
4.0082855
62.433556
coin
92.938835
0.0
43.98483
exit
86.05887
0.0
52.05919
over
    """






def Dc_test(module_path,index_start = 0,index_end = None):
    
    start = index_start
    if index_end == None:
        end = len(DC_dic)
    else:
        end = index_end
    i = 0
    for dc_key in DC_dic:
        if i >=start and i <= end:
            dc = DC_dic[dc_key]
            env = make_vec_env(lambda:MRL_env(dc = dc))
            save_path = 'test_log/ctrlDc/'+dc_key
            os.makedirs(save_path, exist_ok=True)
            test_tool.ctrl_test( module_path,env,1000,save_path)
        i+=1





DC_dic = {

"B01":[90,-90,-90],
"B02":[60,-90,-90],
"B03":[30,-90,-90],
"C01":[-90,90,-90],
"C02":[-90,60,-90],
"C03":[-90,30,-90],
"R01":[-90,-90,90],
"R02":[-90,-90,60],
"R03":[-90,-90,30],

}


train_ctrl()

#procces_statistic_testlog.statistic_testlog("test_log/ctrl_log/action_log.csv")
#Dc_test('trained_modules/ctrl/normal_bestB2')







