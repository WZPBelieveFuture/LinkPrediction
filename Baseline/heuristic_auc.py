import numpy as np
import pandas as pd
import networkx as nx
import argparse,random
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Heuristic Method')
parser.add_argument('--network_name', type=str, default='twitter',
                    help='network name')
parser.add_argument('--method_name', type=str, default='CN',
                    help='method name')
parser.add_argument('--degree_method', type=str, default='outdegree',
                    help='degree method')
parser.add_argument('--iter', type=int, default=1,
                    help='iter')
args = parser.parse_args()

ITER_N = 800000

def Construct_Network():
    userdata = pd.read_csv('../../origin_data/%s/userList.txt'%args.network_name, header=None)
    edgedata = pd.read_csv('../../origin_data/%s/followgraph.txt'%args.network_name, header=None)
    user_list = list(userdata.iloc[:][0])
    G1 = nx.DiGraph()
    G1.add_nodes_from(user_list)
    G1.add_edges_from(zip(edgedata.iloc[:][0], edgedata.iloc[:][1]))
    return G1, user_list, edgedata

def Cal_NonExistEdge(g, users, test_number):
    noexist_number = 0
    non_test_dict = {}
    while(noexist_number < test_number):
        left = random.sample(users, 1)[0]
        right = random.sample(users, 1)[0]
        one_order = list(g.successors(left))
        if len(one_order) == 0 or right not in list(g.successors(left)):
            if left not in non_test_dict:
                non_test_dict[left] = []
                non_test_dict[left].append(right)
                noexist_number += 1
            else:
                if right not in non_test_dict[left]:
                    non_test_dict[left].append(right)
                    noexist_number += 1
    no_left = []
    no_right = []
    for key, value in non_test_dict.items():
        for _ in value:
            no_left.append(key)
            no_right.append(_)
    d_test = pd.DataFrame({"a": no_left, "b": no_right})
    d_test.to_csv('../Temp/Part_NonExistEdge_%s_%s.txt'%(args.network_name, args.iter), index=False, header=False)

def Construct_Training(user_list, edgedata, ratio):
    g = nx.DiGraph()
    d_train, d_test = train_test_split(edgedata, test_size=ratio)
    d_test.to_csv('../Temp/test_%s_data_%s.txt'%(args.network_name, args.iter), index=False, header=False)
    g.add_nodes_from(user_list)
    g.add_edges_from(zip(d_train.iloc[:][0], d_train.iloc[:][1]))
    return g, len(d_test)

def CN(node1, node2, g):

    return len(set(g.successors(node1)) & set(g.predecessors(node2)))

def Salton(node1, node2, g):
    temp = 0
    if args.degree_method == 'indegree':
        temp = len(set(g.predecessors(node1))) * len(set(g.predecessors(node2)))
    elif args.degree_method == 'outdegree':
        temp = len(set(g.successors(node1))) * len(set(g.successors(node2)))
    elif args.degree_method == 'degree':
        temp = len(set(nx.all_neighbors(g, node1))) * len(set(nx.all_neighbors(g, node2)))

    if temp == 0:
        return 0
    else:
        return CN(node1, node2, g) / np.sqrt(temp)

def Jaccard(node1, node2, g):

    temp = len(set(g.successors(node1)) | set(g.predecessors(node2)))
    if temp == 0:
        return 0
    else:
        return CN(node1, node2, g) / temp

def Sorenson(node1, node2, g):
    temp = 0
    if args.degree_method == 'indegree':
        temp = len(set(g.predecessors(node1))) + len(set(g.predecessors(node2)))
    elif args.degree_method == 'outdegree':
        temp = len(set(g.successors(node1))) + len(set(g.successors(node2)))
    elif args.degree_method == 'degree':
        temp = len(set(nx.all_neighbors(g, node1))) + len(set(nx.all_neighbors(g, node2)))
    if temp == 0:
        return 0
    else:
        return 2 * CN(node1, node2, g) / temp

def HPI(node1, node2, g):

    temp = 0
    if args.degree_method == 'indegree':
        temp = np.min([len(set(g.predecessors(node1))), len(set(g.predecessors(node2)))])
    elif args.degree_method == 'outdegree':
        temp = np.min([len(set(g.successors(node1))), len(set(g.successors(node2)))])
    elif args.degree_method == 'degree':
        temp = np.min([len(set(nx.all_neighbors(g, node1))), len(set(nx.all_neighbors(g, node2)))])
    if temp == 0:
        return 0
    else:
        return CN(node1, node2, g) / temp

def HDI(node1, node2, g):
    temp = 0
    if args.degree_method == 'indegree':
        temp = np.max([len(set(g.predecessors(node1))), len(set(g.predecessors(node2)))])
    elif args.degree_method == 'outdegree':
        temp = np.max([len(set(g.successors(node1))), len(set(g.successors(node2)))])
    elif args.degree_method == 'degree':
        temp = np.max([len(set(nx.all_neighbors(g, node1))), len(set(nx.all_neighbors(g, node2)))])
    if temp == 0:
        return 0
    else:
        return CN(node1, node2, g) / temp

def LHN_I(node1, node2, g):
    temp = 0
    if args.degree_method == 'indegree':
        temp = len(set(g.predecessors(node1))) * len(set(g.predecessors(node2)))
    elif args.degree_method == 'outdegree':
        temp = len(set(g.successors(node1))) * len(set(g.successors(node2)))
    elif args.degree_method == 'degree':
        temp = len(set(nx.all_neighbors(g, node1))) * len(set(nx.all_neighbors(g, node2)))
    if temp == 0:
        return 0
    else:
        return CN(node1, node2, g) / temp

def AA(node1, node2, g):

    if args.degree_method == 'indegree':
        return sum([1 / np.log(g.in_degree(cn)) if g.in_degree(cn) != 0 else 0 for cn in list(set(g.successors(node1)) & set(g.predecessors(node2)))])
    elif args.degree_method == 'outdegree':
        return sum([1 / np.log(g.out_degree(cn)) if g.out_degree(cn) != 0 else 0 for cn in list(set(g.successors(node1)) & set(g.predecessors(node2)))])
    elif args.degree_method == 'degree':
        return sum([1 / np.log(g.degree(cn)) if g.degree(cn) != 0 else 0 for cn in list(set(g.successors(node1)) & set(g.predecessors(node2)))])

def RA(node1, node2, g):
    if args.degree_method == 'indegree':
        return sum([1 / g.in_degree(cn) if g.in_degree(cn) != 0 else 0 for cn in list(set(g.successors(node1)) & set(g.predecessors(node2)))])
    elif args.degree_method == 'outdegree':
        return sum([1 / g.out_degree(cn) if g.out_degree(cn) != 0 else 0 for cn in list(set(g.successors(node1)) & set(g.predecessors(node2)))])
    elif args.degree_method == 'degree':
        return sum([1 / g.degree(cn) if g.degree(cn) != 0 else 0 for cn in list(set(g.successors(node1)) & set(g.predecessors(node2)))])

def auc(node1, node2, g):
    if args.method_name == 'CN':
        return CN(node1, node2, g)
    elif args.method_name == 'Salton':
        return Salton(node1, node2, g)
    elif args.method_name == 'Jaccard':
        return Jaccard(node1, node2, g)
    elif args.method_name == 'Sorenson':
        return Sorenson(node1, node2, g)
    elif args.method_name == 'HPI':
        return HPI(node1, node2, g)
    elif args.method_name == 'HDI':
        return HDI(node1, node2, g)
    elif args.method_name == 'LHN_I':
        return LHN_I(node1, node2, g)
    elif args.method_name == 'AA':
        return AA(node1, node2, g)
    elif args.method_name == 'RA':
        return RA(node1, node2, g)

def Cal_AUC(test_exist_data, test_no_exist_data, g):
    score = 0
    for _ in range(ITER_N):
        idx1 = random.randint(0, len(test_exist_data) - 1)
        test_scores = auc(test_exist_data.iloc[idx1][0], test_exist_data.iloc[idx1][1], g)
        idx2 = random.randint(0, len(test_no_exist_data) - 1)
        non_score = auc(test_no_exist_data.iloc[idx2][0], test_no_exist_data.iloc[idx2][1], g)
        if test_scores > non_score:
            score += 1
        elif test_scores == non_score:
            score += 0.5
    auc_score = score / ITER_N
    pd.DataFrame({'auc': [auc_score]}).to_csv('../result/%s/%s/%s.txt'%(args.network_name, args.degree_method, args.method_name), header=None, index=None, mode='a')

if __name__== '__main__':
    g1, user_list, edgedata = Construct_Network()
    g2, test_length = Construct_Training(user_list, edgedata, 0.1)
    Cal_NonExistEdge(g1, user_list, test_length)
    test_exist_data = pd.read_csv('../Temp/test_%s_data_%s.txt' % (args.network_name, args.iter), header=None)
    test_no_exist_data = pd.read_csv('../Temp/Part_NonExistEdge_%s_%s.txt' % (args.network_name, args.iter), header=None)
    for name in ['Salton', 'Jaccard', 'Sorenson', 'HPI', 'HDI', 'LHN_I', 'CN', 'AA', 'RA']:
        args.method_name = name
        for degree in ['indegree', 'outdegree', 'degree']:
            args.degree_method = degree
            Cal_AUC(test_exist_data, test_no_exist_data, g2)
            
