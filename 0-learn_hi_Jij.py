"""
input: 
    1. "raw_data/450-510-global_state_day_time_0.25_0.09.csv"

output: 
    1. "result_data/MPF-ising-f-0.25-q01-0.09-450-510.json" (h_list; J_list)
"""



import pandas as pd
import numpy as np
from coniii import *
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# train_model
def train_data(f_array):

    dict_HJ_result = {}

    try:
    ##################训练###################################
        solver = MPF(f_array)
        solver.solve()

        list_HJ_result = solver.multipliers.tolist() 

        print('q_01%s-f%s' % (q_01, f))
        print(list_HJ_result)
        dict_HJ_result['%s-%s' % (q_01, f)] = list_HJ_result

    except ValueError:
        print("q_01%s-f=%s-Jacobian_problem" % (q_01, f))


    res_file = open("result_data/aaa-MPF-ising-f-%s-q01-%s-%s-%s.json" % (f, q_01,time_start,time_end), 'w')
    res_file.write(json.dumps(dict_HJ_result) + '\n')
    res_file.close()



q_01 = 0.09
f = 0.25


x_start=1
y_start=1
x_end=6
y_end=5
cell_index_list = []
for x_i in range(x_start, x_end):
    for y_i in range(y_start, y_end):
        cell_index_list.append('%s-%s' % (x_i, y_i))
n = len(cell_index_list)

time_start=450  # 7:30am
time_end=510  # 8:30am


if __name__=='__main__':

    df = pd.read_csv("raw_data/450-510-global_state_day_time_0.25_0.09.csv")
    df['str'] = df['str'].apply(lambda x: str(x).zfill(n))

    f_str_list = list(df['str'])

    # generate data for learn
    f_array = np.array([list(s) for s in f_str_list], dtype=int)
    
    # change 0 to -1
    f_array[f_array==0] = -1
    f_array_learn = f_array.astype(np.float64)

    # train_model
    train_data(f_array_learn)

