import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import networkx as nx
from itertools import product 
import functools
import operator

x_start=1
y_start=1
x_end=6
y_end=5
cell_index_list = []
for x_i in range(x_start, x_end):
    for y_i in range(y_start, y_end):
        cell_index_list.append('%s-%s' % (x_i, y_i))


# each_global_state_probability
def return_ising_model_p(dict_e_result):
    dict_p_result = {}
    sum_e = 0
    for key in dict_e_result.keys():
        key_e = np.exp(-dict_e_result[key])
        sum_e += key_e

    for key in dict_e_result.keys():
        key_p = float(np.exp(-dict_e_result[key])/sum_e)
        dict_p_result[key] = key_p  
    return dict_p_result

def return_distance(str1, str2):
    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1
    return distance

# data_analysis
def return_data_num1_p(f_str, n):
    dict_result = {}
    for n in range(n+1):
        dict_result[n] = 0
    for f_str_i in list(set(f_str)): 
        p = float(f_str.count(f_str_i)/len(f_str))
        num1 = f_str_i.count('1')
        dict_result[num1] += p
    return dict_result

def return_data_lcc_p(f_str, dict_lcc, n):
    dict_result = {}
    for n in range(n+1):
        dict_result[n] = 0
    for f_str_i in list(set(f_str)):  
        p = float(f_str.count(f_str_i)/len(f_str))
        lcc = dict_lcc[f_str_i]
        dict_result[lcc] += p
    return dict_result

# model_analysis
def return_model_lcc_p(n, dict_p_result, dict_lcc_result):
    dict_lcc_p_result = {}
    for i in range(n+1):
        dict_lcc_p_result[i] = 0
    for key in dict_p_result.keys():
        key_p = dict_p_result[key]
        lcc_size_p = dict_lcc_result[key]  # lcc_size
        dict_p_result[key] = key_p
        dict_lcc_p_result[lcc_size_p] += key_p
    return dict_lcc_p_result

def return_model_num1_p(n, dict_p_result):
    # 计算相空间所有状态的出现频率
    dict_num1_p_result = {}
    for i in range(n+1):
        dict_num1_p_result[i] = 0
    for key in dict_p_result.keys():
        key_p = dict_p_result[key]
        num1 = key.count('1') # 该状态的num1
        dict_p_result[key] = key_p
        dict_num1_p_result[num1] += key_p
    return dict_num1_p_result


def calculate_R_2(list1, list2):
    # change list to numpy
    x = np.array(list1)
    y = np.array(list2)
    # mean
    y_mean = np.mean(y)
    # SS_tot and SS_res
    SS_tot = np.sum((y - y_mean) ** 2)
    SS_res = np.sum((y - x) ** 2)
    # R²
    R_squared = 1 - (SS_res / SS_tot) 
    return R_squared


# state transition network
def return_change_str(str_old, change_index):
    new_str_list = []
    for i in range(len(str_old)):
        value = int(str_old[i])
        if i in change_index:
            value = 1- value
        
        new_str_list.append(value)
    
    new_str = ''.join([str(i) for i in new_str_list])
    return new_str


class DisconnectivityGraph(object):
    """
    输入：能量字典和局部最小点

    网络1：随着能量阈值不断增加，符合能量条件的节点不断增加
    网络2（非连通图）：叶子节点代表局部最小点；交叉节点代表能量阈值（在这个阈值下，能连通的局部最小点）；连边代表聚合过程；方向：叶节点指向交叉节点
        每个节点的两个性质：状态列表；该点对应的能量（局部最小点或者能量阈值）

    步骤：
    从能量最低点开始，不断升高能量值；
    将小于该能量值的节点加入路径或连通子图判断网络中
    ①判断是否增加了新的局部最小点
        如果有，则加入新节点，将其自身作为性质1，将其自身能量作为性质2
    ②判断连通子图中是否新加入了交叉节点
        1. 计算网络弱连通子图
        2. 对于所有没有子节点的节点：根据其状态列表，判断是否存在两个及以上节点处于同一连通子图
        3. 如果存在，则新加入节点，把同一个连通子图中的状态列表合并成为其性质1，将该能量阈值视为其性质2
    
    直到存在某个节点的状态列表中包含了全部局部最小点
    """
    def __init__(self, local_minima_list, dict_e_result, delta_E):
        self.local_minima_list = local_minima_list  # 根据局部最小点所处的
        self.dict_e_result = dict_e_result
        self.dict_local_minima_index = {}
        for i in range(len(local_minima_list)):
            self.dict_local_minima_index[local_minima_list[i]] = i 
        
        self.delta_E = delta_E
    
    def return_distance(self, str1, str2):
        distance = 0
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                distance += 1
        return distance
    
    def return_test_loop(self, dict_DG_node):

        test_loop = 0
        if len(dict_DG_node.keys()) != 0:
            for key in dict_DG_node.keys():
                value = len(dict_DG_node[key]['state_list'])
                if value > test_loop:
                    test_loop = value
        
        return test_loop

    def return_new_edge_list(self,list_test_node_new, list_test_node_old):  # 微观状态之间的边
        new_edge_list = []
        for i in range(len(list_test_node_new)-1):
            for j in range(i+1, len(list_test_node_new)):
                if self.return_distance(list_test_node_new[i], list_test_node_new[j]) == 1:
                    new_edge_list.append((list_test_node_new[i], list_test_node_new[j]))
        if len(list_test_node_old) != 0:
            for node_new in list_test_node_new:
                for node_old in list_test_node_old:
                    if self.return_distance(node_new, node_old) == 1:
                        new_edge_list.append((node_new, node_old)) 
        return new_edge_list
    
    def return_new_DG(self, test_G_new, DG, dict_DG_node, E, node_id):
        new_DG = nx.DiGraph()
        new_DG_node_list = list(DG.nodes())
        new_DG_edge_list = list(DG.edges())

        # 输出DG中出度为0的节点
        list_no_out_nodes = []
        for node in DG.nodes():
            if DG.out_degree(node) == 0:
                list_no_out_nodes.append(node)
        
        list_no_out_nodes_test = list_no_out_nodes
        print(dict_DG_node)
        # 对于no_out的节点，统计其状态集合中的某个节点是否与其他no_out中的节点处于同一连通子图中

        list_in_one_component = []

        while len(list_no_out_nodes_test) > 1:
            no_out_node_i = list_no_out_nodes_test[0]
            no_out_node_i_one_state = dict_DG_node[no_out_node_i]['state_list'][0]

            list_in_one_component_i = [no_out_node_i]

            for j in range(1, len(list_no_out_nodes_test)):
                # no_out_node_j = list_no_out_nodes[j]
                no_out_node_j = list_no_out_nodes_test[j]
                no_out_node_j_one_state = dict_DG_node[no_out_node_j]['state_list'][0]

                if nx.has_path(test_G_new, no_out_node_i_one_state, no_out_node_j_one_state):
                    list_in_one_component_i.append(no_out_node_j)
            
            list_no_out_nodes_test = [i for i in list_no_out_nodes_test if i not in list_in_one_component_i]

            list_in_one_component.append(list_in_one_component_i)
            
        test_aaa = 0
        for component in list_in_one_component:
            if len(component) > 1:
                new_node_state_list = []
                sum_x = 0
                for component_i in component:
                    new_node_state_list.extend(dict_DG_node[component_i]['state_list'])
                    sum_x += dict_DG_node[component_i]['x']
                
                dict_DG_node[node_id] = {}
                dict_DG_node[node_id]['E_y'] = E
                dict_DG_node[node_id]['state_list'] = new_node_state_list
                dict_DG_node[node_id]['x'] = float(sum_x/len(component))

                new_DG_node_list.append(node_id)

                for component_i in component:
                    new_DG_edge_list.append((component_i, node_id))

                node_id+=1
                test_aaa+=1
        
        if test_aaa == 0:
            return 0
        else:
            new_DG.add_nodes_from(new_DG_node_list)
            new_DG.add_edges_from(new_DG_edge_list)
            return new_DG, dict_DG_node, node_id

    def return_final_value(self):

        # 构建DG的节点字典  应该是三个性质，第三个性质是横坐标
        dict_DG_node = {}
        # 构建DG
        DG = nx.DiGraph()
        # 构建判断连通子图的网络
        list_test_node = []
        list_test_edge = []

        # 已经加入DG中的local minima
        local_minima_list_in = []
        # 还未加入DG中的local minima
        local_minima_list_out = self.local_minima_list

        E = min(list(self.dict_e_result.values()))-0.1

        node_id = 0  # 根据加入的先后顺序命名

        test_loop = 0

        while test_loop < len(self.local_minima_list):  # 只要还有未加入DG的local minima，就继续加入
            print(node_id)

            list_test_node_new = [key for key in list(self.dict_e_result.keys()) if ((self.dict_e_result[key] > E-self.delta_E) & (self.dict_e_result[key] <= E))]

            if len(list_test_node_new) > 0: 

                local_minima_list_now_in = []
                # 第一步：判断是否有新的local minima加入
                for local_minima in local_minima_list_out:
                    if ((self.dict_e_result[local_minima] > E-self.delta_E) & (self.dict_e_result[local_minima] <= E)):  
                        local_minima_list_now_in.append(node_id)
                        dict_DG_node[node_id] = {}
                        dict_DG_node[node_id]['E_y'] = self.dict_e_result[local_minima]
                        dict_DG_node[node_id]['state_list'] = [local_minima]
                        dict_DG_node[node_id]['x'] = self.dict_local_minima_index[local_minima]
                        node_id += 1
                # 更新还未加入DG中的local minima
                if len(local_minima_list_now_in) > 0:
                    local_minima_list_in.extend(local_minima_list_now_in)
                    DG.add_nodes_from(local_minima_list_now_in)
                    local_minima_list_out = [i for i in self.local_minima_list if i not in local_minima_list_in]


                # 第二步：判断这些节点的状态列表中，是否在同一连通子图(需要构建判断连通子图的网络)，有的话需要加入虚拟节点
                
                # 更新网络中的节点和连边
                
                list_test_edge_new = self.return_new_edge_list(list_test_node_new, list_test_node)

                list_test_node.extend(list_test_node_new)

                if len(list_test_edge_new) > 0:
                    list_test_edge.extend(list_test_edge_new)  
                    # 新增加边了，那么就要从DG中看出度为0的节点能否合并了，也就是判断是否要给DG添加虚拟节点和连边
                
                    test_G_new = nx.Graph()
                    test_G_new.add_nodes_from(list_test_node)
                    test_G_new.add_edges_from(list_test_edge)

                    value = self.return_new_DG(test_G_new, DG, dict_DG_node, E, node_id)
                    if value != 0:
                        DG, dict_DG_node, node_id = value

            E += self.delta_E
            test_loop = self.return_test_loop(dict_DG_node)
        

        print(DG.nodes())
        print(DG.edges())
        print(dict_DG_node)
        print(node_id)

        return DG, dict_DG_node
    

    def return_DG_plot(self, DG, dict_DG_node):

        node_pos = {}
        for node in dict_DG_node.keys():
            node_pos[node] = (dict_DG_node[node]['x'], dict_DG_node[node]['E_y'])
        

        labels = {}
        for node in DG.nodes():
            labels[node] = 'merge'
            if len(dict_DG_node[node]['state_list']) == 1:
                labels[node] = dict_DG_node[node]['state_list'][0]


        plt.figure()
        nx.draw_networkx_nodes(DG, pos=node_pos, node_size = 200, alpha=1)
        nx.draw_networkx_edges(DG, pos=node_pos, edge_color='blue', width=1.5, arrows=True, alpha=0.8)
        nx.draw_networkx_labels(DG, pos=node_pos, labels=labels)
        plt.plot([0, 0], [-19, -9])
        plt.show()






### fig
def draw_arrow_real2(pos_,G_time_real, alpha, color, width):
    edge_pos = np.asarray([(pos_[e[0]], pos_[e[1]]) for e in G_time_real.edges()])
    ax = plt.gca()
    arrow_collection = []
    for i, (src, dst) in enumerate(edge_pos):
        x1, y1 = src
        x2, y2 = dst
        shrink_source = 0  # space from source to tail
        shrink_target = 0  # space from  head to target
        arrow = mpl.patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle='-',
            shrinkA=shrink_source,
            shrinkB=shrink_target,
            alpha=alpha,
            color=color[i],
            linewidth=width[i],
            connectionstyle='arc3, rad = 0.3',
            #linestyle=linestyle,
            mutation_scale=20,
            zorder=1,
        )  # arrows go behind nodes
        arrow_collection.append(arrow)
        ax.add_patch(arrow)


def two_bar_plot(ax, list_x, list_y1, list_y2, title, xmin=0, xmax=1):
    # 设置字体为新罗马
    plt.rcParams['font.family'] = 'Times New Roman'

    # 绘制 step 线图
    ax.step(list_x, list_y1, where='mid', label='Data', color='green', linewidth=3)
    ax.step(list_x, list_y2, where='mid', label='Model', color='orange', linewidth=3)
    ax.set_xlim(xmin, xmax)
    
    # 设置刻度大小和粗细
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, direction='in')  # 主刻度朝内
    ax.tick_params(axis='both', which='minor', labelsize=10, width=1, direction='in')  # 次刻度朝内

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

    ax.legend(prop={'weight': 'bold'})


def draw_pdf(y_list,ax, bin_num, bar_color, title, range, alpha=1, edgecolor=None, y_log=False):
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    # Histogram:
    # Bin it
    n, bin_edges = np.histogram(y_list, bins=bin_num, range=range)
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n/float(n.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
    print(bin_middles)
    print(bin_probability)
    ax.bar(bin_middles, bin_probability, edgecolor=edgecolor, color=bar_color, width=bin_width, alpha=alpha)
    ax.set_title(label=title)
    if y_log:
        ax.set_yscale('log')

def draw_pdf_n(y_list,ax, bin_num, bar_color, title, range, alpha=1, edgecolor=None, y_log=False, p=False):
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    # Histogram:
    # Bin it
    n, bin_edges = np.histogram(y_list, bins=bin_num, range=range)
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n/float(n.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
    if p:
        ax.bar(bin_middles, bin_probability, edgecolor=edgecolor, color=bar_color, width=bin_width, alpha=alpha)
    else:
        ax.bar(bin_middles, n, edgecolor=edgecolor, color=bar_color, width=bin_width, alpha=alpha)

    if y_log:
        ax.set_yscale('log')
    ax.set_title(label=title)

def double_bar(x, y1, y2, color1='red', color2='green'):
    import matplotlib.pyplot as plt
    import numpy as np

    # # X轴数据
    # x = np.arange(5)  # 5个数据点

    # # 两个Y轴数据
    # y1 = np.random.randint(1, 10, size=5)
    # y2 = np.random.randint(1, 10, size=5)

    # 创建画布和子图
    fig, ax1 = plt.subplots()

    # 绘制第一个Y轴的柱状图
    ax1.bar(x - 0.2, y1, width=0.4, color=color1, align='center', label='Y1', edgecolor='black')
    ax1.bar(x + 0.2, y2, width=0.4, color=color2, align='center', label='Y2', edgecolor='black')

    # 添加标题和标签
    plt.title('Double Bar Chart with Different Y Axes')
    plt.xlabel('X')

    # 显示图形
    plt.show()

# 拟合
def return_Line_fit_list(x, y):
    from scipy.optimize import curve_fit

    def Line(x, a, b):
        return a*x + b

    param_l, param_cov_l = curve_fit(Line, np.array(x), np.array(y))
    ans_y = Line(np.array(x), *param_l)

    return ans_y

def scatter_plot(list_x, list_y, ax, color, label, fit=True):
    ax.scatter(list_x, list_y,label=label, color=color)
    if fit:
        list_y_ans = return_Line_fit_list(list_x, list_y)
        ax.plot(list_x, list_y_ans, color)
