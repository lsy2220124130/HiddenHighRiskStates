"""
input: 
    1. "result_data/dict_e_result.json" (each_global_state_energy)
    2. "raw_data/450-510-global_state_day_time_0.25_0.09.csv"
    3. "result_data/each_global_state_lcc_size.json"

output: 
    1. Fig.2b(energy distribution of states)
    2. Fig.2e (G_p distribution)
"""
import pandas as pd
import matplotlib.pyplot as plt
from function import draw_pdf_n, return_ising_model_p
import json
import os
import numpy as np
import seaborn as sns


os.chdir(os.path.dirname(os.path.abspath(__file__)))


def e_distribution_two_2(dict_e_result, f_str):
    list_all_e = list(dict_e_result.values())
    range_aaa = (min(list_all_e), max(list_all_e))
    print(range_aaa)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # ax1 = ax.twinx()
    bin_num=50
    bar_color = 'black'
    title = 'all_state_e'
    draw_pdf_n(list_all_e, ax, bin_num, bar_color, title, range_aaa, alpha=0.4,y_log=True) 

    dict_f_str_e_result = {}
    for key in dict_e_result.keys():
        if f_str.count(key) >0:
            dict_f_str_e_result[key] = dict_e_result[key]
    list_data_e = list(dict_f_str_e_result.values())

    bar_color = 'white'
    title = 'data_state_e'
    draw_pdf_n(list_data_e, ax, bin_num, bar_color, title, range_aaa, alpha=1, edgecolor='black',y_log=True)

    ax.set_ylim(0.1, 200000)

    # 设置字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'
    for ax in [ax]:
        # 设置刻度大小和粗细
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, direction='in')  # 主刻度朝内
        ax.tick_params(axis='both', which='minor', labelsize=10, width=1, direction='in')  # 次刻度朝内

        # 设置边框宽度
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # 设置边框宽度
        
        # 加粗刻度标签
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    plt.show()

def heatmap_plot_G_p(dict_p_result, dict_lcc, f_str=[]):
    import seaborn as sns 
    import matplotlib.pyplot as plt
    if len(f_str) == 0:
        dict_p_result_aaa = dict_p_result
    if len(f_str) != 0:
        dict_p_result_aaa = {}
        for key in dict_p_result.keys():
            if key not in f_str:
                dict_p_result_aaa[key] = dict_p_result[key]
    list_model_p = [-np.log10(x) for x in list(dict_p_result_aaa.values())]
    list_model_G = list([float(dict_lcc[key]) for key in dict_p_result_aaa.keys()])
    list_aaa = list(zip(list_model_p, list_model_G))

    array_p_G = np.zeros((12, 11))
    for p_aaa_i in range(12):
        p_aaa_min = p_aaa_i*2+1
        p_aaa_max = p_aaa_i*2 + 3
        for G_aaa_j in range(11):
            G_aaa_min = G_aaa_j * 2
            G_aaa_max = G_aaa_j * 2 + 2

            list_aaa_p_G = [x for x in list_aaa if (x[0] >= p_aaa_min) & (x[0] < p_aaa_max) & (x[1] >= G_aaa_min) & (x[1] < G_aaa_max)]
            array_p_G[p_aaa_i][G_aaa_j] = len(list_aaa_p_G)

    array_p_G_log_10 = np.log10(np.where(array_p_G == 0, 0.1, array_p_G))
    print(array_p_G_log_10)

    x_list = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    y_list = ['10e-0~10e-3']

    for p_aaa_i in range(1, 12):
        y_list.append('10e-%s~10e-%s' % (p_aaa_i*2+1, p_aaa_i*2 + 3))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(array_p_G_log_10, xticklabels=x_list, yticklabels=y_list, square=False, ax=ax)
    # 设置字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'
    for ax in [ax]:
        # 设置刻度大小和粗细
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, direction='in')  # 主刻度朝内
        ax.tick_params(axis='both', which='minor', labelsize=10, width=1, direction='in')  # 次刻度朝内

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
time_start=450
time_end=510

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

    df = pd.read_csv("raw_data/450-510-global_state_day_time_0.25_0.09.csv")
    df['str'] = df['str'].apply(lambda x: str(x).zfill(n))
    f_str = list(df['str'])

    with open("result_data/dict_e_result.json") as f_node:
        dict_e_result = json.load(f_node)

    # Fig.2b(energy distribution of states) 
    e_distribution_two_2(dict_e_result, f_str)

    # Fig.2e (G_p distribution)
    with open("result_data/each_global_state_lcc_size.json", 'rb') as f_dict_lcc:
        dict_lcc = json.load(f_dict_lcc)
    dict_p_result = return_ising_model_p(dict_e_result)
    heatmap_plot_G_p(dict_p_result, dict_lcc, f_str=[])