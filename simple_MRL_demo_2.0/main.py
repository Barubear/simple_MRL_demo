
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

import train

from Envs.P1_Env import p1_Env
from Envs.P2_Env import p2_Env
from Envs.P3_Env import p3_Env
from Envs.P4_Env import p4_Env
from Envs.Dctest_env import Dctest_env
from Envs.Modular_test_env import Modular_test_env
from Envs.MRL_env import MRL_env
from Envs.MRL_test_env import MRL_test_env
import test_tool
import procces_statistic_testlog
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
        batch_size=1024,  # 批大小
        n_steps=1024,  # 步数
        n_epochs=16,  # 训练次数
       
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=2),  # LSTM 设置
        verbose=1,
        )
    else :
        model = re_tarin_model
        model.set_env(env)
    
    train.train(model,env,times,save_path,log_path,test_function=test_function,test_times=test_time,test_only=testonly)


def train_navi_p1():
    save_path = 'simple_MRL_demo_2.0/trained_modules/p1/normal_best02'
    log_path = 'logs/p1_log'
    env = make_vec_env("p1_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,2000000,testonly=True)
    """
state value
52.019547
-83.966324
11.514604
step
2.2258999999999998
0.995
    """




def train_navi_p2():
    save_path = 'simple_MRL_demo_2.0/trained_modules/p2/normal_best03'
    log_path = 'logs/p2_log'
    env = make_vec_env("p2_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,1000000,testonly=True)

    """
state value
39.802402
-42.067253
11.286243
step
3.0606
0.957
    """



def train_navi_p3():
    save_path = 'simple_MRL_demo_2.0/trained_modules/p3/normal_best02'
    log_path = 'logs/p3_log'
    env = make_vec_env("p3_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,1000000,testonly=True)

"""
state value
46.291714
-26.392736
27.115408
step
0.955
1.0
"""
def train_navi_p4():
    save_path = 'simple_MRL_demo_2.0/trained_modules/p4/normal_best'
    log_path = 'simple_MRL_demo_2.0/logs/p4_log'
    env = make_vec_env("p4_Env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,2000000,testonly=True)
    """
    104.0913
-108.98085
22.560017
step
21.716
0.996
    """


def train_ctrl():
    save_path = 'simple_MRL_demo_2.0/trained_modules/ctrlchange/normal_best02'
    log_path = 'simple_MRL_demo_2.0/logs/ctrl_log'
    env = make_vec_env("MRL_env-v0",monitor_dir=log_path)
    Moduletrain(save_path,log_path,env,2000000,test_function=test_tool.ctrl_test)
    """
94.362
4.015213
59.31661
coin
94.22733
8.588558
56.477364
exit
95.376656
3.6020875
62.54081
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
            save_path = 'simple_MRL_demo_2.0/test_log\change-dc/'+dc_key
            os.makedirs(save_path, exist_ok=True)
            test_tool.ctrl_test( module_path,env,1000,save_path)
        i+=1

def Dc_statistic(index_start = 0,index_end = None):
    start = index_start
    if index_end == None:
        end = len(DC_dic)
    else:
        end = index_end
    i = 0
    for dc_key in DC_dic:
        
        if i >=start and i <= end:
            load_path = "simple_MRL_demo_2.0/test_log\ctrl-org\changeDC/"+dc_key+"/action_log.csv"
            save_path = "simple_MRL_demo_2.0\statistic_log\ctrl-org\changeDC/"+dc_key+"/"
            os.makedirs(save_path, exist_ok=True)
            procces_statistic_testlog.statistic_testlog(load_path, exNum=dc_key,save_path=save_path)
        i+=1


        
        
        

#train_ctrl()


#procces_statistic_testlog.statistic_testlog("simple_MRL_demo_2.0/test_log/ctrl-battle-change/action_log.csv",save_path="simple_MRL_demo_2.0/statistic_log/ctrl-change/rog02")
DC_dic = {

"B01":[90,-90,-90],
"B02":[60,-90,-90],
"B03":[30,-90,-90],
"B04":[90,0,0],
"B05":[80,0,0],
"B06":[70,0,0],
"B07":[60,0,0],
"B08":[50,0,0],
"B09":[40,0,0],
"B10":[30,0,0],
"B11":[20,0,0],
"B12":[10,0,0],
"B13":[-50,0,0],
"B14":[-40,0,0],
"B15":[-30,0,0],
"B16":[-20,0,0],
"B17":[-10,0,0],

"C01":[-90,90,-90],
"C02":[-90,60,-90],
"C03":[-90,30,-90],
"C04":[0,90,0],
"C05":[0,80,0],
"C06":[0,70,0],
"C07":[0,60,0],
"C08":[0,50,0],
"C09":[0,40,0],
"C10":[0,30,0],
"C11":[0,20,0],
"C12":[0,10,0],
"C13":[0,-50,0],
"C14":[0,-40,0],
"C15":[0,-30,0],
"C16":[0,-20,0],
"C17":[0,-10,0],

"R01":[-90,-90,90],
"R02":[-90,-90,60],
"R03":[-90,-90,30],
"R04":[0,0,90],
"R05":[0,0,80],
"R06":[0,0,70],
"R07":[0,0,60],
"R08":[0,0,50],
"R09":[0,0,40],
"R10":[0,0,30],
"R11":[0,0,20],
"R12":[0,0,10],
"R13":[0,0,-50],
"R14":[0,0,-40],
"R15":[0,0,-30],
"R16":[0,0,-20],
"R17":[0,0,-10],

}

#Dc_test("simple_MRL_demo_2.0/trained_modules/ctrlchange/normal_best02",index_start=3,index_end=16)

#Dc_statistic(3,16)
#procces_statistic_testlog.comparative_analysis_sw(DC_dic,index_start=4,index_end=16,comparative_index=0)
#train_navi_p4()

procces_statistic_testlog.Relationship("comparative_analysis.csv",0,1,"battle change")





