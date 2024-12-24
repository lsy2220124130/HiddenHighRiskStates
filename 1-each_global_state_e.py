"""
input: 
    1. "result_data/MPF-ising-f-0.25-q01-0.09-450-510.json" (h_list; J_list)

output: 
    1. "result_data/dict_e_result.json" (each_global_state_energy)
"""

from itertools import product 
import json
import os
import functools
import operator

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def tup_to_str(tup1):
    ''' 
    tuple2str
    '''
    tup2 = tuple(map(str, tup1))
    str_aaa = functools.reduce(operator.add, (tup2))
    return str_aaa

# each_global_state_energy
def return_ising_model_e(Jij, H_i, n):
    """
    each_global_state_energy
    """
    inputs = list(product([-1, 1], repeat=n))
    inputs_01 = list(product([0, 1], repeat=n))

    dict_e_result = {}
    for input_k_i in range(len(inputs)):
        input_k = inputs[input_k_i]
        input_k_01 = inputs_01[input_k_i]

        E_k_1 = 0  # first_part
        J_ij_index = 0
        for i in range(n-1):
            for j in range(i+1,n):
                E_k_1 -= input_k[i]*input_k[j]*Jij[J_ij_index]
                J_ij_index += 1
        E_k_2 = 0  # second_part
        for state_i in range(n):
            E_k_2 -= input_k[state_i]*H_i[state_i]
        dict_e_result[tup_to_str(input_k_01)] = E_k_1+E_k_2
    return dict_e_result


q_01 = 0.09
f = 0.25
time_start=450  # 7:30am
time_end=510  # 8:30am
n = 20

if __name__=='__main__':
    with open("result_data/MPF-ising-f-%s-q01-%s-%s-%s.json" % (f, q_01,time_start,time_end), 'rb') as f_hj:
        js_hj = json.load(f_hj)
        list_HJ = js_hj['%s-%s' % (q_01, f)]
        h_list = list_HJ[0:n]
        J_list = list_HJ[n:]

    dict_e_result = return_ising_model_e(J_list, h_list, n)

    res_file = open('result_data/dict_e_result.json', 'w')
    res_file.write(json.dumps(dict_e_result) + '\n')
    res_file.close()

