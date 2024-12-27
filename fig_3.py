"""
input: 
    1. "result_data/dict_e_result.json" (each_global_state_energy)
    2. "result_data/each_global_state_lcc_size.json"

output: 
    1. local minima
    2. fig3a: each state position according to local minima (as origin input)
    3. fig3b: DisconnectivityGraph
    4. fig.3c: accessibility of local minima
    5. fig.3d: distribution of distance between local minima
"""
from tqdm import tqdm
from itertools import combinations
import function
from function import return_change_str, return_ising_model_p, return_distance, double_bar
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

# each state position according to local minima (as origin input)
def G_landscape(G, dict_p_result_big_p):

    # 从每个无入度的节点出发，使其出来的边只保留一条
    def return_new_G(G):
        # 就是每个点只有一个出边(保留p最大的target) 
        new_G = nx.DiGraph()
        list_edge = []
        for node in G.nodes():
            out_neighbors = list(G.successors(node))
            if len(out_neighbors) == 1:
                list_edge.append((node, out_neighbors[0]))
            elif len(out_neighbors) > 1:  # 保留p最大的target
                max_p = 0
                max_out_nei = None
                for out_neighbor in out_neighbors:
                    p = dict_p_result_big_p[out_neighbor]
                    if p > max_p:
                        max_p = p
                        max_out_nei = out_neighbor
                list_edge.append((node, max_out_nei))
        new_G.add_edges_from(list_edge)
        return new_G
    
    def return_new_G_pos(G, dict_center_pos):
        import math
        
        def count_ancestors(G, node, memo={}):
            if node in memo:
                return memo[node]
            ancestors_count = 0
            for parent in G.predecessors(node):
                ancestors_count += 1 + count_ancestors(G, parent, memo)
            memo[node] = ancestors_count
            return ancestors_count

        def layered_layout_one_component(one_component, center, center_pos):
            pos = {center: np.array(center_pos)}
            layers = {}
            for node in one_component.nodes():
                if node != center:
                    layers[node] = nx.shortest_path_length(one_component, node, center)
            print(layers)
            max_distance = max(layers.values())

            # 第一层节点：计算祖先数量并排序
            first_layer_nodes = [node for node, dist in layers.items() if dist == 1]
            ancestors_counts = {node: count_ancestors(one_component, node) for node in first_layer_nodes}
            sorted_nodes = sorted(first_layer_nodes, key=lambda node: ancestors_counts[node], reverse=True)

            # 最多祖先的两个节点放在圆的对立面
            max1_node, max2_node = sorted_nodes[0], sorted_nodes[1]

            radius = 65  # 第一层半径固定
            pos[max1_node] = np.array([center_pos[0] + radius * np.cos(0), center_pos[1] + radius * np.sin(0)])
            pos[max2_node] = np.array([center_pos[0] + radius * np.cos(math.pi), center_pos[1] + radius * np.sin(math.pi)])

            # 交错放置剩余节点
            remaining_nodes = sorted_nodes[2:]
            # 两组间交错，每组包括节点数量多的和少的
            group1, group2 = [], []
            toggle = True
            for i in range(len(remaining_nodes)):
                node = remaining_nodes[i]
                if toggle:
                    group1.append(node)
                else:
                    group2.append(node)
                toggle = not toggle
            
            # print('aaaaaaaaaaaaa')
            # print(len(remaining_nodes))
            # print(len(group1)+len(group2))

            # 确定每组节点的角度
            group1_angles = np.linspace(math.pi / len(group1), math.pi, len(group1), endpoint=False)
            group2_angles = np.linspace(math.pi, 2 * math.pi - math.pi / len(group2), len(group2), endpoint=False)

            for i in range(len(group1)):
                node = group1[i]
                pos[node] = np.array([center_pos[0] + radius * np.cos(group1_angles[i]), center_pos[1] + radius * np.sin(group1_angles[i])])

            for i in range(len(group2)):
                node = group2[i]
                pos[node] = np.array([center_pos[0] + radius * np.cos(group2_angles[i]), center_pos[1] + radius * np.sin(group2_angles[i])])

            # 从第二层开始，节点尝试与其子节点的角度保持一致，允许小的偏移
            for distance in range(2, max_distance + 1):
                nodes_at_distance = [node for node, dist in layers.items() if dist == distance]
                for node in nodes_at_distance:
                    if one_component.successors(node):  # 确保节点有子节点
                        child = next(one_component.successors(node))  # 获取唯一的子节点
                        child_pos = pos[child]
                        angle = math.atan2(child_pos[1] - center_pos[1], child_pos[0] - center_pos[0])
                        offset = random.uniform(-0.1, 0.1)  # 稍微分散一点点

                        radius = distance * 40  # 根据层级距离设置半径

                        pos[node] = np.array([
                            center_pos[0] + radius * np.cos(angle + offset),
                            center_pos[1] + radius * np.sin(angle + offset)
                        ])

            return pos


        # 找出弱连通子图
        components = list(nx.weakly_connected_components(G))
        # 创建并打印所有弱连通组件的子图
        subgraphs = []
        for component in components:
            # 创建每个组件的子图
            subgraph = G.subgraph(component)
            subgraphs.append(subgraph)
            # 打印子图的节点和边
            # print("Nodes:", subgraph.nodes())
            # print("Edges:", subgraph.edges())
            print(nx.is_weakly_connected(subgraph))
            # print(subgraph.nodes())

        pos_all = {}
        # 对于每个component
        for subgraph in subgraphs:
            center_list = [n for n in subgraph.nodes() if subgraph.out_degree(n) == 0]
            # print(center_list)
            center = center_list[0]
            center_pos = dict_center_pos[center]
            # 获取分层布局
            pos_one = layered_layout_one_component(subgraph, center, center_pos)
            for key, value in pos_one.items():
                pos_all[key] = value
        
        return pos_all, subgraphs

    
    new_G = return_new_G(G)
    print(new_G.number_of_nodes())

    # set initial minima position based on prog='twopi'
    dict_center_pos = {'11000010011111001100': (0, -200), '11000010011101011110': (-800, 0),'01000010000101001100': (800, 0), 
                       '00000110000101011100': (500, 350), '10000110001101011110': (-400, 300),'00001110000101111100': (0, 200), '11101100011101010111': (-1300, -200)}

    pos_, subgraphs = return_new_G_pos(new_G, dict_center_pos)
    
    # output
    list_result = []
    com_num = 1
    for component in nx.weakly_connected_components(new_G):
        for node in component:
            list_result.append((com_num, node, pos_[node][0], pos_[node][1], dict_e_result[node]))
        com_num += 1
    df_result = pd.DataFrame(list_result, columns=['com_num', 'node', 'pos1', 'pos2', 'e'])
    df_result.to_excel('result_data/for_origin_pos_e_high_p_state_pos_com.xlsx', index=False)

# attractor basin for each local minimum
def return_one_node_graph(G, leap_node_list, f_str=[]):
    """
    从叶节点开始，不断向上找祖先
    """
    def find_ancestors(graph, node, ancestors, visited):
        """
        递归地查找节点的所有祖先
        """
        predecessors = list(graph.predecessors(node))
        if len(predecessors) == 0:
            return ancestors
        else:
            for parent in predecessors:
                if parent not in visited:
                    visited.add(parent)
                    ancestors.append(parent)
                    find_ancestors(graph, parent, ancestors, visited)
        return ancestors
    
    def find_ancestors_path_length(graph, node, ancestors, visited, k=0):
        """
        递归地查找节点的所有祖先；
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
    
    dict_ancestors_model = {}
    for leaf_node in tqdm(leap_node_list):
        ancestors = find_ancestors(G, leaf_node, [], set())
        dict_ancestors_model[leaf_node] = ancestors
    return dict_ancestors_model

# distance between local minima
def distance_distribution_type(leap_node_list_big_basin_green, leap_node_list_big_basin_red):
    y_red_pre = []

    list_aaa = leap_node_list_big_basin_red
    for i in range(len(list_aaa)-1):
        for j in range(i+1, len(list_aaa)):
            dis = return_distance(list_aaa[i], list_aaa[j])
            y_red_pre.append(dis)

    y_green_pre = []
    list_aaa = leap_node_list_big_basin_green
    for i in range(len(list_aaa)-1):
        for j in range(i+1, len(list_aaa)):
            dis = return_distance(list_aaa[i], list_aaa[j])
            y_green_pre.append(dis)

    x_list = list(range(max([max(y_red_pre),max(y_green_pre)])+1))

    y_red = []
    y_green = []
    for i in x_list:
        y_red.append(y_red_pre.count(i))
        y_green.append(y_green_pre.count(i))

    print(x_list)
    print(y_red)
    print(y_green)
    return np.array(x_list), np.array(y_red), np.array(y_green)


n = 20

if __name__=='__main__':
    with open("result_data/dict_e_result.json") as f_node:
        dict_e_result = json.load(f_node)
    dict_p_result = return_ising_model_p(dict_e_result)
    # filter big_p
    dict_p_result_big_p = {}
    for key, value in dict_p_result.items():
        if value > 0.00001:
            dict_p_result_big_p[key] = value
    with open("result_data/each_global_state_lcc_size.json", 'rb') as f_dict_lcc:
        dict_lcc = json.load(f_dict_lcc)
    
    # local minima
    G, sorted_minima = state_transition_net(dict_p_result_big_p, dict_lcc)
    print(sorted_minima)
    """
    由上面的代码可以得到7个local minima 及其对应的lcc:
    [('01000010000101001100', 14), ('00000110000101011100', 13), ('11000010011101011110', 7), ('11000010011111001100', 6), ('00001110000101111100', 5), ('10000110001101011110', 4), ('11101100011101010111', 3)]
    """
    aaa_all = [('01000010000101001100', 14), ('00000110000101011100', 13), ('11000010011101011110', 7), ('11000010011111001100', 6), ('00001110000101111100', 5), ('10000110001101011110', 4), ('11101100011101010111', 3)]
    local_minima_list = [x[0] for x in aaa_all]

    # # fig.3a
    # G_landscape(G, dict_p_result_big_p)

    # # fig.3b
    # class_DG = function.DisconnectivityGraph(local_minima_list, dict_e_result,delta_E=0.1)
    # DG, dict_DG_node = class_DG.return_final_value()
    # class_DG.return_DG_plot(DG, dict_DG_node)

    # fig3c
    # 画吸引子的basin大小等
    dict_ancestors_model = return_one_node_graph(G, local_minima_list)
    x_list = np.array(list(range(len(local_minima_list))))
    y_model_list = np.array([len(dict_ancestors_model[x]) for x in local_minima_list])
    print(x_list)
    print(y_model_list)
    plt.figure()
    plt.bar(x_list, y_model_list, color='blue')
    plt.yscale('log')
    plt.show()

    # fig3d
    leap_node_list_green = [x[0] for x in aaa_all if x[1]>=10]
    leap_node_list_red = [x[0] for x in aaa_all if x[1]<10]
    # # # 补充类型内距离和类型间距离
    x_list, y_inner, y_inter = distance_distribution_type(leap_node_list_green, leap_node_list_red)
    double_bar(x_list, y_inner, y_inter)




