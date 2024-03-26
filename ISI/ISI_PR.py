import pandas as pd
import numpy as np
import networkx as nx
import random
import copy
from sklearn.model_selection import train_test_split
import argparse
parser = argparse.ArgumentParser(description='Link prediction based on the user tweets and network structure')
# general settings
parser.add_argument('--network_name', default=None, help='network name')
parser.add_argument('--iter', type=float, default=1,
                    help='running number')
parser.add_argument('--test_ratio', type=float, default=0.001,
                    help='negative test ratio')
args = parser.parse_args()
p_r_test_edges = {}
p_r_non_edges = {}

length_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def get_testing_edges_for_p_r(test_edges):
    for i in range(len(test_edges)):
        left = test_edges.iloc[i][0]
        if left not in p_r_test_edges.keys():
            p_r_test_edges[left] = []
        p_r_test_edges[left].append(test_edges.iloc[i][1])
    
def get_non_edges_for_p_r(g):
    
    for idx, value in enumerate(p_r_test_edges.keys()):
        non_nodes = set(user_list) - set(p_r_test_edges[value]) - set([value])
        if value in list(nx.nodes(g)):
            non_nodes -= set(g.neighbors(value))
        p_r_non_edges[value] = list(non_nodes)
    
def Construct_Network():
    userdata = pd.read_csv('./origin_data/%s/userList.txt'%args.network_name, header=None)
    edgedata = pd.read_csv('./origin_data/%s/followgraph.txt'%args.network_name, header=None)
    user_list = list(userdata.iloc[:][0])
    G1 = nx.DiGraph()
    G1.add_nodes_from(user_list)
    G1.add_edges_from(zip(edgedata.iloc[:][0], edgedata.iloc[:][1]))
    return G1, user_list, edgedata

def Cal_NonExistEdge(g):
    no_exist_edge = np.array(list(nx.non_edges(g)))
    data = pd.DataFrame({"a": no_exist_edge[:, 0], "b": no_exist_edge[:, 1]})
    d_train,d_test = train_test_split(data, test_size=args.test_ratio)
    d_test.to_csv('./Temp_pr/Part_NonExistEdge_%s_%s.txt'%(args.network_name, args.iter), index=False, header=False)

def Construct_Training(user_list, edgedata):
    g = nx.DiGraph()
    d_train, d_test = train_test_split(edgedata, test_size=0.1)
    d_test.to_csv('./Temp_pr/test_%s_data_%s.txt'%(args.network_name, args.iter), index=False, header=False)
    g.add_nodes_from(user_list)
    g.add_edges_from(zip(d_train.iloc[:][0], d_train.iloc[:][1]))
    return g

def Normalization_User_Feature(user_list):
    user_norm_feature_dict = {}
    data=pd.read_csv('./origin_data/%s/userfeature.txt'%args.network_name, sep='\t', header=None)
    for i in range(len(data)):
        user_feature = np.array(data.loc[i][0:100])
        user_norm_feature_dict[user_list[i]] = user_feature/np.sum(user_feature)
    return user_norm_feature_dict

def Top30_User_Feature(user_list, feature_dict):
    user_norm_feature_dict_30 = {}
    for i in user_list:
        user_feature = np.zeros(100, dtype=np.float)
        user_feature[np.argpartition(feature_dict[i], -30)[-30:]] = feature_dict[i][np.argpartition(feature_dict[i], -30)[-30:]]
        user_norm_feature_dict_30[i] = user_feature/np.sum(user_feature)
    return user_norm_feature_dict_30
    
def Revise_User_Feature(user_list, origin_feature, top30_feature, k, g):
    new_feature_dict = copy.deepcopy(top30_feature)
    for user in user_list:
        if k == 0:
            new_feature_dict[user] = origin_feature[user].copy()
        elif k == 1:
            one_order = list(g.successors(user))
            if len(one_order)!=0:
                for nei in one_order:
                    new_feature_dict[user] += top30_feature[nei].copy()
        elif k == 2:
            q = list(nx.dfs_preorder_nodes(g, source=user, depth_limit=1))
            p = list(nx.dfs_preorder_nodes(g, source=user, depth_limit=2))
            two_order = list(set(p).difference(set(q)))
            if len(two_order)!=0:
                for nei in two_order:
                    new_feature_dict[user] += top30_feature[nei].copy()
        elif k == 3:
            q = list(nx.dfs_preorder_nodes(g, source=user, depth_limit=2))
            p = list(nx.dfs_preorder_nodes(g, source=user, depth_limit=3))
            three_order = list(set(p).difference(set(q)))
            if len(three_order)!=0:
                for nei in three_order:
                    new_feature_dict[user] += top30_feature[nei].copy()
        elif k == 4:
            q = list(nx.dfs_preorder_nodes(g, source=user, depth_limit=3))
            p = list(nx.dfs_preorder_nodes(g, source=user, depth_limit=4))
            four_order = list(set(p).difference(set(q)))
            if len(four_order)!=0:
                for nei in four_order:
                    new_feature_dict[user] += top30_feature[nei].copy()
    return new_feature_dict

def Read_TestData():
    similarity_result = []
    test_data = pd.read_csv('./Temp_pr/test_%s_data_%s.txt'%(args.network_name, args.iter), header=None)
    return test_data

def CosSimility(feature1, feature2):
    a_norm = np.linalg.norm(feature1)
    b_norm = np.linalg.norm(feature2)
    return np.dot(feature1, feature2) / (a_norm*b_norm)

def Cal_Simility(g, user1, user2, rev_vec):
    mips = 0
    mips += CosSimility(rev_vec[user1], rev_vec[user2])
    list1 = list(g.successors(user1))
    list2 = list(g.predecessors(user2))
    list3 = list(set(list1).intersection(set(list2)))
    if len(list3) != 0:
        for _ in list3:
            mips += CosSimility(rev_vec[_], rev_vec[user2])
    return mips

def precise_and_recall_for_one(g, user, rev_vec):

    score_record = []
    for _ in p_r_test_edges[user]:
        score_record.append([1, Cal_Simility(g, user, _, rev_vec)])
    for _ in p_r_non_edges[user]:
        score_record.append([0, Cal_Simility(g, user, _, rev_vec)])
    sorted_score_record = sorted(score_record, key=lambda x: x[1], reverse=True)
    precise_list = []
    for _ in length_list:
        precise_list.append(sum([x[0] for x in sorted_score_record[:_]]) / _)
    recall_list = []
    for _ in length_list:
        recall_list.append(sum([x[0] for x in sorted_score_record[:_]]) / len(p_r_test_edges[user]))
    return precise_list, recall_list

def precise_and_recall(g, test_exist, rev_vector):
    get_testing_edges_for_p_r(test_exist)
    get_non_edges_for_p_r(g)
    precise_ = np.array(np.zeros(len(length_list)))
    recall_ = np.array(np.zeros(len(length_list)))
    nodes_list = [x for x in p_r_test_edges.keys()]
    for counter, _ in enumerate(nodes_list):
        p, r = precise_and_recall_for_one(g, _, rev_vector)
        precise_ += np.array(p)
        recall_ += np.array(r)
    precise_ /= len(nodes_list)
    recall_ /= len(nodes_list)
    fout = open(r'./result/pr/%s/precise_%s.txt'%(args.network_name,args.iter), 'w', encoding='utf-8')
    for _ in zip(length_list, precise_):
        fout.write(str(_[0]) + ' ' + str(_[1]) + '\n')
    fout.close()
    fout = open(r'./result/pr/%s/recall_%s.txt'%(args.network_name,args.iter), 'w', encoding='utf-8')
    for _ in zip(length_list, recall_):
        fout.write(str(_[0]) + ' ' + str(_[1]) + '\n')
    fout.close()

if __name__=='__main__':
    g1, user_list, edgedata = Construct_Network()
    Cal_NonExistEdge(g1)
    g2 = Construct_Training(user_list, edgedata)
    origin_norm_user_feature = Normalization_User_Feature(user_list)
    norm_user_feature_30 = Top30_User_Feature(user_list, origin_norm_user_feature)
    revise_user_feature = Revise_User_Feature(user_list, origin_norm_user_feature, norm_user_feature_30,  1, g2)
    test_data = Read_TestData()
    precise_and_recall(g2, test_data, revise_user_feature)










