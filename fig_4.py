"""
input: 
    1. local minima with large accessibility (obtained by fig. 3c)
    2. "result_data/each_global_state_lcc_size.json"
    3. "raw_data/450-510-global_state_day_time_0.25_0.09.csv"
    4. "result_data/each_global_state_lcc_size.json"

output: 
    1. fig.4b: R distribution for high-p-states with G>=0.5
    2. fig.4c: transition from observed high-p-states with G>=0.5
    3. fig.4d: R distribution for high-p-states with G<0.3
    4. fig.4e: transition from observed high-p-states with G<0.3

middle_result:
    1. 'result_data/high_p_state_to_big_basin_attractor.xlsx' (distance from high_p_states to 5 local minima with large accessibility)

"""

from tqdm import tqdm
from itertools import combinations
import function
from function import return_change_str, return_distance, return_ising_model_p, draw_pdf
import networkx as nx
import json
import os
import numpy as np
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import random
import matplotlib.pyplot as plt


# state transition network and local minima
def return_one_attractor_and_neighbor(state, layer):
    list_state = [state]
    if layer > 0:
        list_state = []
        index_list = list(range(n))
        change_index_list = [list(set_i) for set_i in list(combinations(index_list, layer))]
        for change_index in change_index_list:
            layer_str = return_change_str(state, change_index)
            list_state.append(layer_str)
    return list_state

def state_transition_net(dict_p_result_big_p, dict_lcc):
    """
    edge: energy_source > energy_target
    """
    node_list = list(dict_p_result_big_p.keys())
    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    list_edge = []
    for node_i in tqdm(node_list):
        list_neighbor = return_one_attractor_and_neighbor(node_i, layer=1)
        for node_j in list_neighbor:
            if node_j in node_list:  # i's neighbors in node_list
                if dict_p_result_big_p[node_j] > dict_p_result_big_p[node_i]:  # j with lower energy, then i flow into j
                    list_edge.append((node_i, node_j))
                else:
                    list_edge.append((node_j, node_i))
    
    G.add_edges_from(list_edge)
    # out_degrees
    out_degrees = dict(G.out_degree())
    local_minima = [node for node in G.nodes() if out_degrees[node]==0]
    local_minimal_dict = {}
    for minimum in local_minima:
        local_minimal_dict[minimum] = dict_lcc[minimum]
    sorted_minima = sorted(local_minimal_dict.items(), key=lambda x: x[1], reverse=True)  # sort_minima by lcc
    # print(sorted_minima)

    return G, sorted_minima

def return_one_node_graph_path(G, leap_node_list, f_str=[]):

    def find_ancestors_path_length(graph, node, ancestors, visited, k=0):
        """
        递归地查找节点的所有祖先
        """
        predecessors = list(graph.predecessors(node))
        if len(predecessors) == 0:
            return ancestors
        else:
            k += 1
            for parent in predecessors:
                if parent not in visited:
                    visited.add(parent)
                    ancestors.append((parent, k))
                    find_ancestors_path_length(graph, parent, ancestors, visited, k)
        return ancestors
    
    dict_ancestors_path_length = {}

    for leaf_node in tqdm(leap_node_list):

        ancestors_path_length = find_ancestors_path_length(G, leaf_node, [], set(), k=0)
        dict_ancestors_path_length[leaf_node] = {}
        for aaa in ancestors_path_length:
            dict_ancestors_path_length[leaf_node][aaa[0]] = aaa[1]

    return dict_ancestors_path_length


def high_p_state_to_big_basin_attractor(leap_node_list_big_basin, dict_ancestors_path_length, dict_p_result_big_p):
    """
    distance=100 when there is no path
    """

    # 每个高概率状态到5个低能量吸引子的最短距离（最短距离与汉明距离之间的关系）
    list_result = []
    for high_p_state in tqdm(dict_p_result_big_p.keys()):
        list_one_state = [high_p_state]
        for big_basin_attractor in leap_node_list_big_basin:
            dict_one_ancestor_path_length = dict_ancestors_path_length[big_basin_attractor]
            result = 100
            if high_p_state in dict_one_ancestor_path_length.keys():
                result = dict_one_ancestor_path_length[high_p_state]
            
            # 计算路径的最短距离
            list_one_state.append(result)

            # 计算汉明距离
            list_one_state.append(return_distance(high_p_state, big_basin_attractor))

        list_result.append(list_one_state)

    column_list = ['high_p_state']
    for big_basin_attractor in leap_node_list_big_basin:
        column_list.append('%s_path' % big_basin_attractor)
        column_list.append('%s_distance' % big_basin_attractor)
    df_result = pd.DataFrame(list_result, columns=column_list)

    return df_result

def known_state_transition(f_str_day_time, minute, dict_lcc, lcc_threshold=6):
    """
    flow to dangerous states
    """
    # 计算每个状态在未来给定时间内的状态列表(如果某个状态出现了多次,就将多次的未来状态都列里面)
    dict_f_str_next_state_list = {}
    for kkk in range(len(f_str_day_time)):
        time = f_str_day_time[kkk][1]
        if time < time_end-minute-1:  # 只考虑后面有足够时长的样本
            state = f_str_day_time[kkk][2]
            if state not in list(dict_f_str_next_state_list.keys()):
                dict_f_str_next_state_list[state] = []
            next_state_list = [f_str_day_time[kkk + aaa][2] for aaa in range(1, minute+1)]
            dict_f_str_next_state_list[state].extend(next_state_list)
    dict_less_G_bili = {}
    for key in list(dict_f_str_next_state_list.keys()):
        list_less_G = [x for x in dict_f_str_next_state_list[key] if dict_lcc[x]< lcc_threshold]
        # G小于0.3的比例
        dict_less_G_bili[key] = float(len(list_less_G)/len(dict_f_str_next_state_list[key]))
    return dict_less_G_bili

# fig
def Risk_distribution(list_final_pre):
    # 将状态根据最终落入的吸引子情况分成四类：①同时落入两类吸引子②只落入正常吸引子③只落入危险吸引子④没有 # Supplementary Fig. 4
    y_normal_list = []
    y_hazardous_list = []
    y1 = [x for x in list_final_pre if (x[1] < 100) & (x[2]<100)]
    y2 = [x for x in list_final_pre if (x[1] < 100) & (x[2]==100)]
    y3 = [x for x in list_final_pre if (x[1] == 100) & (x[2]<100)]
    y4 = [x for x in list_final_pre if (x[1] == 100) & (x[2]==100)]
    y_aaa_list = [y1,y2,y3,y4]
    print([len(y) for y in y_aaa_list])
    for y in y_aaa_list:
        y_normal_list.append(len([bbb for bbb in y if bbb[4] >= 10]))
        y_hazardous_list.append(len([bbb for bbb in y if bbb[4] < 10]))
    print(y_normal_list)
    print(y_hazardous_list)

    # 画F(bili)的分布，两类G （fig 4b 和 Supplementary Fig. 6a）
    list_final_big_G_F = [np.log10(x[5]) for x in list_final_pre if x[4]>=10]
    list_final_small_G_F = [np.log10(x[5]) for x in list_final_pre if x[4]<6]
    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    draw_pdf(y_list=list_final_big_G_F,ax=ax1, bin_num=20, bar_color='green', title='F distribution for states with G≥0.5', range=(np.log10(0.009), np.log10(201)), alpha=1, edgecolor=None, y_log=True)
    draw_pdf(y_list=list_final_small_G_F,ax=ax2, bin_num=20, bar_color='red', title='F distribution for states with G<0.3', range=(np.log10(0.009), np.log10(201)), alpha=1, edgecolor=None, y_log=True)
    plt.show()

def verify_risk_indicator(list_final_pre):
    # risk
    list_final_find_hidden_high_risk = [x for x in list_final_pre if (x[3]==0) and (x[4]>=10)]
    print(sorted(list_final_find_hidden_high_risk, key=lambda x:x[5], reverse=True)[0:10])
    print(sorted(list_final_find_hidden_high_risk, key=lambda x:x[5], reverse=False)[0:10])

    #  resilience
    list_final_find_hidden_resilience = [x for x in list_final_pre if (x[3]==0) and (x[4]<6)]
    print(sorted(list_final_find_hidden_resilience, key=lambda x:x[5], reverse=False)[0:10])
    print(sorted([x for x in list_final_pre if (x[4] >= 10) & (x[3]==1)], key=lambda y:y[5], reverse=True))


    # # # verify_risk_indicator by dynamics of observed states
    # risk
    list_final_big_F_risk = [x for x in list_final_pre if (x[5]>=10) and (x[3]==1) and (x[4]>=10)]
    list_final_small_F_risk = [x for x in list_final_pre if (x[5]<1) and (x[3]==1) and (x[4]>=10)]

    # resilience
    list_final_big_F_resilience = [x for x in list_final_pre if (x[5]>=10) and (x[3]==1) and (x[4]<6)]
    list_final_small_F_resilience = [x for x in list_final_pre if (x[5]<1) and (x[3]==1) and (x[4]<6)]

    time_list = []
    y_big_F_risk = []
    y_std_big_F_risk = []
    y_small_F_risk = []
    y_std_small_F_risk = []

    y_big_F_resilience = []
    y_std_big_F_resilience = []
    y_small_F_resilience = []
    y_std_small_F_resilience = []


    for time in tqdm(range(1, 30)):
        time_list.append(time)

        dict_data_transition_risk = known_state_transition(f_str_day_time, time, dict_lcc, lcc_threshold=10)
        dict_data_transition_resilience = known_state_transition(f_str_day_time, time, dict_lcc, lcc_threshold=6)

        list_data_transtion_green_result = []
        for aaa in list_final_big_F_risk:
            if aaa[0] in dict_data_transition_risk.keys():
                list_data_transtion_green_result.append(dict_data_transition_risk[aaa[0]])
        y_big_F_risk.append(np.mean(list_data_transtion_green_result))
        y_std_big_F_risk.append(np.std(list_data_transtion_green_result, ddof=1)/np.sqrt(len(list_data_transtion_green_result)))

        list_data_transtion_green_result = []
        for aaa in list_final_small_F_risk:
            if aaa[0] in dict_data_transition_risk.keys():
                list_data_transtion_green_result.append(dict_data_transition_risk[aaa[0]])
        y_small_F_risk.append(np.mean(list_data_transtion_green_result))
        y_std_small_F_risk.append(np.std(list_data_transtion_green_result, ddof=1)/np.sqrt(len(list_data_transtion_green_result)))


        list_data_transtion_green_result = []
        for aaa in list_final_big_F_resilience:
            if aaa[0] in dict_data_transition_resilience.keys():
                list_data_transtion_green_result.append(dict_data_transition_resilience[aaa[0]])
        y_big_F_resilience.append(np.mean(list_data_transtion_green_result))
        y_std_big_F_resilience.append(np.std(list_data_transtion_green_result, ddof=1)/np.sqrt(len(list_data_transtion_green_result)))

        list_data_transtion_green_result = []
        for aaa in list_final_small_F_resilience:
            if aaa[0] in dict_data_transition_resilience.keys():
                list_data_transtion_green_result.append(dict_data_transition_resilience[aaa[0]])
        y_small_F_resilience.append(np.mean(list_data_transtion_green_result))
        y_std_small_F_resilience.append(np.std(list_data_transtion_green_result, ddof=1)/np.sqrt(len(list_data_transtion_green_result)))

    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.errorbar(time_list, y_big_F_risk, yerr=y_std_big_F_risk, fmt='-o', label='big_F', color='b')   
    ax1.errorbar(time_list, y_small_F_risk, yerr=y_std_small_F_risk, fmt='-o', label='small_F', color='orange')  

    ax2.errorbar(time_list, y_big_F_resilience, yerr=y_std_big_F_resilience, fmt='-o', label='big_F', color='b')   
    ax2.errorbar(time_list, y_small_F_resilience, yerr=y_std_small_F_resilience, fmt='-o', label='small_F', color='orange') 

    plt.legend()
    plt.show()


time_start=450
time_end=510
n = 20
aaa_big_basin = [('01000010000101001100', 14), ('00000110000101011100', 13), ('11000010011101011110', 7), ('11000010011111001100', 6), ('10000110001101011110', 4)] 

if __name__=='__main__':
    # with open("result_data/dict_e_result.json") as f_node:
    #     dict_e_result = json.load(f_node)
    # dict_p_result = return_ising_model_p(dict_e_result)
    # # filter big_p
    # dict_p_result_big_p = {}
    # for key, value in dict_p_result.items():
    #     if value > 0.00001:
    #         dict_p_result_big_p[key] = value
    # with open("result_data/each_global_state_lcc_size.json", 'rb') as f_dict_lcc:
    #     dict_lcc = json.load(f_dict_lcc)

    # # local minima
    # G, sorted_minima = state_transition_net(dict_p_result_big_p, dict_lcc)

    # # fig4  distance from high_p_states to 5 local minima with large accessibility
    # leap_node_list_all = [x[0] for x in aaa_big_basin]
    # dict_ancestors_path_length = return_one_node_graph_path(G, leap_node_list_all, f_str=[])
    # df_result = high_p_state_to_big_basin_attractor(leap_node_list_all, dict_ancestors_path_length, dict_p_result_big_p)
    # df_result.to_excel('result_data/high_p_state_to_big_basin_attractor.xlsx', index=False)

    df_result = pd.read_excel('result_data/high_p_state_to_big_basin_attractor.xlsx')
    list_node = [str(x).zfill(20) for x in list(df_result['high_p_state'])]

    df_bili_f_time = pd.read_csv("raw_data/450-510-global_state_day_time_0.25_0.09.csv")
    df_bili_f_time['str'] = df_bili_f_time['str'].apply(lambda x: str(x).zfill(n))
    f_str = list(df_bili_f_time['str'])
    f_str_day_time = list(zip(list(df_bili_f_time['day']), list(df_bili_f_time['time']), f_str))

    with open("result_data/each_global_state_lcc_size.json", 'rb') as f_dict_lcc:
        dict_lcc = json.load(f_dict_lcc)

    # 第一步，到达两类高可达minima的最短距离：list_final_pre
    normal_columns = ['01000010000101001100_path', '00000110000101011100_path']
    hazardous_columns = ['11000010011101011110_path', '11000010011111001100_path', '10000110001101011110_path']
    # 找出每行在这三列中的最小值
    min_values_normal = df_result[normal_columns].min(axis=1)
    min_values_hazardous = df_result[hazardous_columns].min(axis=1)

    # 将最小值转换为列表
    min_values_normal_list = min_values_normal.tolist()
    min_values_hazardous_list = min_values_hazardous.tolist()
    bili_list = [float(min_values_normal_list[i]/min_values_hazardous_list[i]) for i in range(len(min_values_normal_list))]
    list_str = []
    for node in list_node:
        str_value = 0
        if node in f_str:
            str_value = 1
        list_str.append(str_value)
    list_lcc = [dict_lcc[node] for node in list_node]
    list_final_pre = list(zip(list_node, min_values_normal_list, min_values_hazardous_list, list_str, list_lcc, bili_list))

    # fig 4b and fig 4d
    Risk_distribution(list_final_pre)

    # fig 4c and fig 4e
    verify_risk_indicator(list_final_pre)

