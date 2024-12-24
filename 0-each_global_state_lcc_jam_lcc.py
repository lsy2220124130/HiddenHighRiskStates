"""
input: 
    1. "raw_data/hex_topology_inter_box.xlsx"

output: 
    1. "result_data/each_global_state_lcc_size.json"
    2. "result_data/each_global_state_jam_lcc_size.json"
"""


from itertools import product 
import functools
import operator
import pandas as pd
import networkx as nx
from tqdm import tqdm
import json

"""
0:free; 1:jam
"""


def tup_to_str(tup1):
    ''' 
    change tuple to str
    '''
    tup2 = tuple(map(str, tup1))
    str_aaa = functools.reduce(operator.add, (tup2))
    return str_aaa

def return_result(dict_cell_state):
    df_cell_inter = pd.read_excel("raw_data/hex_topology_inter_box.xlsx")
    G_cell = nx.from_pandas_edgelist(df_cell_inter, 'cell_index_source', 'cell_index_target', create_using=nx.DiGraph())
    G_cell_remove = nx.from_pandas_edgelist(df_cell_inter, 'cell_index_source', 'cell_index_target', create_using=nx.DiGraph())


    remove_node_list = []
    for node in G_cell.nodes():
        if dict_cell_state[node] == 1:
            remove_node_list.append(node)
    if len(remove_node_list) == n:
        value = 0
    else:
        # strong_lcc_list
        G_cell_remove.remove_nodes_from(remove_node_list)
        scc = list(nx.strongly_connected_components(G_cell_remove))
        # max_strong_lcc
        largest_scc = max(scc, key=len)
        value = len(largest_scc)
    return value



def return_result_jam(dict_cell_state):
    df_cell_inter = pd.read_excel("raw_data/hex_topology_inter_box.xlsx")
    G_cell = nx.from_pandas_edgelist(df_cell_inter, 'cell_index_source', 'cell_index_target', create_using=nx.DiGraph())
    G_cell_remove = nx.from_pandas_edgelist(df_cell_inter, 'cell_index_source', 'cell_index_target', create_using=nx.DiGraph())

    remove_node_list = []
    for node in G_cell.nodes():
        if dict_cell_state[node] != 1:
            remove_node_list.append(node)

    if len(remove_node_list) == n:
        value = 0
    else:
        # weak_lcc_list
        G_cell_remove.remove_nodes_from(remove_node_list)
        scc = list(nx.weakly_connected_components(G_cell_remove))
        # max_weak_lcc
        largest_scc = max(scc, key=len)
        value = len(largest_scc)
    return value



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
    
    inputs_01 = list(product([0, 1], repeat=n))

    dict_lcc_result = {}
    dict_jam_lcc_result = {}

    for input_k_i in tqdm(inputs_01):
        input_k_i_str = tup_to_str(input_k_i)
        dict_cell_state = {}
        for i in range(len(input_k_i_str)):
            if input_k_i_str[i] == '1':
                dict_cell_state[cell_index_list[i]] = 1
            else:
                dict_cell_state[cell_index_list[i]] = -1

        len_lcc_i = return_result(dict_cell_state)
        len_jam_lcc_i = return_result_jam(dict_cell_state)

        dict_lcc_result[input_k_i_str] = len_lcc_i
        dict_jam_lcc_result[input_k_i_str] = len_jam_lcc_i



    res_file = open("result_data/each_global_state_lcc_size.json", 'w')
    res_file.write(json.dumps(dict_lcc_result) + '\n')
    res_file.close()

    res_file = open("result_data/each_global_state_jam_lcc_size.json", 'w')
    res_file.write(json.dumps(dict_jam_lcc_result) + '\n')
    res_file.close()
