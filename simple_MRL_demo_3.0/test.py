
import numpy as np
import csv

path="test_log\ctrl_log\state_value_Vector.csv"

yes = 0
total = 0
with open(path,'r') as f:
    reader = csv.reader(f)

    for row in reader:
        total+=1
        b = float(row[0])
        c = float(row[1])
        r = float(row[2])

        max_index = -1

        if b > c and b >= r:
            max_index = 0
        elif c > r:
            max_index = 1
        else:
            max_index = 2


        action = int(row[3][1])

        if action == max_index:
            yes+=1
    


print(yes/total)
