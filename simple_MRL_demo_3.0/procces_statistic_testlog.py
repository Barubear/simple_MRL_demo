import matplotlib.pyplot as plt
import csv
import test_tool
import numpy as np
import os
# 数据

def read_test_log_file(path):

    log=[0]*7
    hp =[]
    with open(path,'r') as f:
        reader = csv.reader(f)

        for row in reader:
            logs = row[0]
            hp.append(int(row[1]))
            for a in logs:
                if a.isdigit():
                    ia = int(a)
                    log[ia] +=1
    mean_hp = np.array(hp).mean()
    over_log = log[0:3]
    action_log = log[3:6]
    return over_log,action_log,mean_hp 


    
def draw_test_log_file(labels,datas,titel,colors = ['gold', 'lightcoral', 'lightskyblue'],explode_index = None,save_path = None):

    d_len = len(datas)

    explode = [0]*d_len  # 突出显示某一块
    if explode_index!=None:
        explode[explode_index] = 0.1

    plt.pie(datas, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=140)

    # 添加图例，右上角显示每个标签对应的色块
    plt.legend(labels, title=titel, loc="upper right", bbox_to_anchor=(1.15, 1))


    # 设置长宽比例相等，确保饼图是圆的
    plt.axis('equal')  
    
    plt.savefig(save_path+titel+'_pie_chart.png', format='png', dpi=300)
    # 显示图表
    #plt.show()



def data_to_PCT(data):
    total = sum(data)
    pct = [(i / total) * 100 for i in data]
    return pct




def statistic_testlog(loadPath,exNum = "ctrl", save_path = 'statistic_log/'):
    title = exNum
    over_log,action_log,mean_hp = read_test_log_file(loadPath)
    os.makedirs(save_path, exist_ok=True)
    draw_test_log_file(['fight','get coin','recover'],action_log,title +' action log',save_path=save_path)
    pct_over_log =data_to_PCT(over_log)
    pct_action_log = data_to_PCT(action_log)
    total_log = [pct_over_log,pct_action_log,mean_hp]
    
    save= save_path+exNum+'.csv'
    test_tool.write_log(save,total_log)
    

