
import torch


def scale_to_range(data,old_min, old_max , new_min=0, new_max=100):

        if data >= old_max:
            data = old_max
        elif data <= old_min:
            data = old_min
       
        scaled_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

        return scaled_data


test_arry = [5.545,10.319,35.397]
    
e = scale_to_range(test_arry[0], -90,80)
c = scale_to_range(test_arry[1], -50,45)
r = scale_to_range(test_arry[2], -30,50)

new_arry = [e+50,c,r]
print(new_arry)
same_scale_stateValue = torch.tensor(new_arry)
same_scale_stateValue = torch.softmax(same_scale_stateValue ,dim= 0)
print(same_scale_stateValue)
