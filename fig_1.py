"""
input: 
    1. "result_data/MPF-ising-f-0.25-q01-0.09-450-510.json" (h_list; J_list)
    2. "result_data/dict_e_result.json" (each_global_state_energy)
    3. "raw_data/450-510-global_state_day_time_0.25_0.09.csv"
    4. "result_data/each_global_state_lcc_size.json"
    5. "result_data/each_global_state_jam_lcc_size.json"

output: 
    1. Fig.1d (interaction_network)
    2. Fig.1e (micro and macro_level_model_fitting_performance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import json
import os
from function import draw_arrow_real2, return_ising_model_p, return_data_lcc_p, return_model_lcc_p, calculate_R_2, return_ising_model_moment, return_data_two_moment,scatter_plot

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

# 验证模型  ①两个moment
def draw_two_moments(dict_p_result, f_array_learn):
    list_first, list_second = return_ising_model_moment(dict_p_result, n)
    # 数据里的两个moment
    mean_spins, mean_interactions = return_data_two_moment(f_array_learn)

    print(len(list_first))
    print(len(mean_spins))
    print(len(list_second))
    print(len(mean_interactions))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    scatter_plot(mean_spins, list_first, ax1, color='b', label='first_moment') 
    ax2 = fig.add_subplot(1, 2, 2)
    scatter_plot(mean_interactions, list_second, ax2, color='b', label='second_moment')

    plt.rcParams['font.family'] = 'Times New Roman'
    for ax in [ax1, ax2]:
        # 设置刻度大小和粗细
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, direction='in')  # 主刻度朝内
        # ax.tick_params(axis='both', which='minor', labelsize=10, width=1, direction='in')  # 次刻度朝内

        # 减少刻度标签数量
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))  # 横坐标减少到5个刻度
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))  # 纵坐标减少到5个刻度

        # 设置边框宽度
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # 设置边框宽度
        
        # 加粗刻度标签
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.show()


def G_distribution_verify_up_down(f_str, dict_p_result):
    with open("result_data/each_global_state_lcc_size.json", 'rb') as f_dict_lcc:
        dict_lcc = json.load(f_dict_lcc)
    dict_data_lcc_p = return_data_lcc_p(f_str, dict_lcc, n)
    dict_model_lcc_p = return_model_lcc_p(n, dict_p_result, dict_lcc)

    list1 = list(dict_data_lcc_p.values())
    list2 = list(dict_model_lcc_p.values())

    R_2 = calculate_R_2(list1, list2)
    print(R_2)

    fig = plt.figure()
    # 设置字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.bar([i/20 for i in list(dict_data_lcc_p.keys())], list(dict_data_lcc_p.values()), color='green', width=0.04)
    ax2.bar([i/20 for i in list(dict_model_lcc_p.keys())], list(dict_model_lcc_p.values()), color='orange', width=0.04)


    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)

    # 设置字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'


    for ax in [ax1, ax2]:
    
        # 设置刻度大小和粗细
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, direction='in')  # 主刻度朝内
        # ax.tick_params(axis='both', which='minor', labelsize=10, width=1, direction='in')  # 次刻度朝内

        # 减少刻度标签数量
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))  # 横坐标减少到5个刻度
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))  # 纵坐标减少到5个刻度

        # 设置边框宽度
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # 设置边框宽度
        
        # 加粗刻度标签
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.show()



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

    # generate data for learn
    f_array = np.array([list(s) for s in f_str], dtype=int)
    
    # change 0 to -1
    f_array[f_array==0] = -1
    f_array_learn = f_array.astype(np.float64)

    # Fig. 1e: 2moments
    draw_two_moments(dict_p_result, f_array_learn)

    G_distribution_verify_up_down(f_str, dict_p_result)


