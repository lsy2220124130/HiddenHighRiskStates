"""
input: 
    1. "result_data/MPF-ising-f-0.25-q01-0.09-450-510.json" (h_list; J_list)
    2. "result_data/dict_e_result.json" (each_global_state_energy)
    3. "raw_data/450-510-global_state_day_time_0.25_0.09.csv"
    4. "result_data/each_global_state_lcc_size.json"
    5. "result_data/each_global_state_jam_lcc_size.json"

output: 
    1. Fig.1d (interaction_network)
    2. Fig.1e (macro_level_model_fitting_performance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import json
import os
from function import draw_arrow_real2, return_ising_model_p, return_data_lcc_p, return_data_num1_p, return_model_lcc_p, return_model_num1_p, calculate_R_2, two_bar_plot

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def return_interaction_network(Jij, thre_j=0.5):

    with open("raw_data\dict_cell_geo.json", 'rb') as f_node:
        pos_ = json.load(f_node)

    # graph
    G_cell = nx.DiGraph()
    G_cell.add_nodes_from(cell_index_list)

    # # node_color   
    # node_color_list = []
    # for i in range(n):
    #     if hi[i] > 0:
    #         node_color = 'orange'
    #     else:
    #         node_color = 'blue'
    #     node_color_list.append(node_color)
            
    # edge_color
    edge_color_list = []
    edge_width_list = []

    k = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if abs(Jij[k]) >= thre_j:
                G_cell.add_edge(cell_index_list[i], cell_index_list[j])
                edge_width_list.append(abs(Jij[k])*5)
                if Jij[k] > 0: 
                    edge_color_list.append('green')
                else:
                    edge_color_list.append('purple')

            k += 1

    plt.figure(figsize=(10.5, 8))
    nx.draw_networkx_nodes(G_cell, pos=pos_, node_size = 6000,  node_shape='h', node_color='None', edgecolors='black', linewidths=2, alpha=1)

    draw_arrow_real2(pos_,G_cell, alpha=1, color=edge_color_list, width=edge_width_list)
    
    plt.scatter([-4,-4, 4,4], [1,-5,1,-5])
    plt.show()


def G_distribution_verify(ax, f_str, dict_p_result, title='G_free'):
    with open("result_data/each_global_state_lcc_size.json", 'rb') as f_dict_lcc:
        dict_lcc = json.load(f_dict_lcc)

    dict_data_lcc_p = return_data_lcc_p(f_str, dict_lcc, n)
    dict_model_lcc_p = return_model_lcc_p(n, dict_p_result, dict_lcc)

    list1 = list(dict_data_lcc_p.values())
    list2 = list(dict_model_lcc_p.values())

    R_2 = calculate_R_2(list1, list2)
    print(R_2)

    two_bar_plot(ax, [i/20 for i in list(dict_data_lcc_p.keys())], list(dict_data_lcc_p.values()), list(dict_model_lcc_p.values()), title, xmin=0, xmax=1)

    df_result = pd.DataFrame({'x': [i/20 for i in list(dict_data_lcc_p.keys())], 'y_data': list(dict_data_lcc_p.values()), 'y_model': list(dict_model_lcc_p.values())})
    return df_result, R_2

def G_jam_distribution_verify(ax, f_str, dict_p_result, title='G_jam'):
    with open("result_data/each_global_state_jam_lcc_size.json", 'rb') as f_dict_lcc:
        dict_lcc = json.load(f_dict_lcc)
    
    dict_data_lcc_p = return_data_lcc_p(f_str, dict_lcc, n)
    dict_model_lcc_p = return_model_lcc_p(n, dict_p_result, dict_lcc)

    list1 = list(dict_data_lcc_p.values())
    list2 = list(dict_model_lcc_p.values())

    R_2 = calculate_R_2(list1, list2)
    print(R_2)

    two_bar_plot(ax, [i/20 for i in list(dict_data_lcc_p.keys())], list(dict_data_lcc_p.values()), list(dict_model_lcc_p.values()), title, xmin=0, xmax=1)
    df_result = pd.DataFrame({'x': list(dict_data_lcc_p.keys()), 'y_data': list(dict_data_lcc_p.values()), 'y_model': list(dict_model_lcc_p.values())})
    return df_result, R_2

def num_distribution_verify(ax, f_str, dict_p_result, title='num'):
    # 统计数据中不同1的个数的占比

    dict_data_num1_p = return_data_num1_p(f_str, n)

    # 统计模型中不同1的个数占比
    dict_model_num1_p = return_model_num1_p(n, dict_p_result)
    # print(dict_data_num1_p.values())
    # print(dict_model_num1_p.values())

    R_2 = calculate_R_2(list(dict_data_num1_p.values()), list(dict_model_num1_p.values()))
    print(R_2)

    two_bar_plot(ax, [i/20 for i in list(dict_data_num1_p.keys())], list(dict_data_num1_p.values()), list(dict_model_num1_p.values()), title, xmin=0, xmax=1)
    df_result = pd.DataFrame({'x': list(dict_data_num1_p.keys()), 'y_data': list(dict_data_num1_p.values()), 'y_model': list(dict_model_num1_p.values())})
    return df_result, R_2


q_01 = 0.09
f = 0.25
time_start=450  # 7:30am
time_end=510  # 8:30am

x_start=1
y_start=1
x_end=6
y_end=5
cell_index_list = []
for x_i in range(x_start, x_end):
    for y_i in range(y_start, y_end):
        cell_index_list.append('%s-%s' % (x_i, y_i))

n = len(cell_index_list)


if __name__=='__main__':

    with open("result_data/MPF-ising-f-%s-q01-%s-%s-%s.json" % (f, q_01,time_start,time_end), 'rb') as f_hj:
        js_hj = json.load(f_hj)
        list_HJ = js_hj['%s-%s' % (q_01, f)]
        h_list = list_HJ[0:n]
        J_list = list_HJ[n:]

    # Fig.1d
    return_interaction_network(J_list, thre_j=0.5)

    # Fig. 1e
    with open("result_data/dict_e_result.json") as f_node:
        dict_e_result = json.load(f_node)
    dict_p_result = return_ising_model_p(dict_e_result)

    df = pd.read_csv("raw_data/450-510-global_state_day_time_0.25_0.09.csv")
    df['str'] = df['str'].apply(lambda x: str(x).zfill(n))
    f_str = list(df['str'])

    # Fig. 1e: G_free
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    df_G_free, R_2 = G_distribution_verify(ax, f_str, dict_p_result)
    plt.show()

    # Fig. 1e: G_jam
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    df_G_jam, R_2 = G_jam_distribution_verify(ax, f_str, dict_p_result)
    plt.show()

    # Fig. 1e: num
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    df_num, R_2 = num_distribution_verify(ax, f_str, dict_p_result)
    plt.show()

