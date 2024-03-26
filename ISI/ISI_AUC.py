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
parser.add_argument('--S', type=int, default=30,
                    help='top S')
args = parser.parse_args()

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
    d_test.to_csv('./Temp/Part_NonExistEdge_%s_%s.txt'%(args.network_name, args.iter), index=False, header=False)

def Construct_Training(user_list, edgedata):
    g = nx.DiGraph()
    d_train, d_test = train_test_split(edgedata, test_size=0.1)
    d_test.to_csv('./Temp/test_%s_data_%s.txt'%(args.network_name, args.iter), index=False, header=False)
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

def CosSimility(feature1, feature2):
    a_norm = np.linalg.norm(feature1)
    b_norm = np.linalg.norm(feature2)
    return np.dot(feature1, feature2) / (a_norm*b_norm)

def Test_ExistData(g, user_feature):
    similarity_result = []
    test_data = pd.read_csv('./Temp/test_%s_data_%s.txt'%(args.network_name, args.iter), header=None)
    for i in range(len(test_data)):
        mips = 0.0
        a = test_data.iloc[i][0]
        b = test_data.iloc[i][1]
        mips += CosSimility(user_feature[a], user_feature[b])
        list1 = list(g.successors(a))
        list2 = list(g.predecessors(b))
        list3 = list(set(list1).intersection(set(list2)))
        if len(list3) != 0:
            for _ in list3:
                mips += CosSimility(user_feature[_], user_feature[b])
        similarity_result.append(mips)
    return similarity_result

def Test_NoExistData(g, user_feature):
    similarity_result = []
    test_data = pd.read_csv('./Temp/Part_NonExistEdge_%s_%s.txt'%(args.network_name, args.iter), header=None)
    for i in range(len(test_data)):
        mips = 0.0
        a = test_data.iloc[i][0]
        b = test_data.iloc[i][1]
        mips += CosSimility(user_feature[a], user_feature[b])
        list1 = list(g.successors(a))
        list2 = list(g.predecessors(b))
        list3 = list(set(list1).intersection(set(list2)))
        if len(list3) != 0:
            for _ in list3:
                mips += CosSimility(user_feature[_], user_feature[b])
        similarity_result.append(mips)
    return similarity_result

def Cal_AUC(data_score1, data_score2):
    num1 = 0
    num2 = 0
    RunTime = 200000
    for i in range(RunTime):
        x = random.sample(data_score1, 1)
        y = random.sample(data_score2, 1)
        if(x > y):
            num1 += 1
        elif(x == y):
            num2 += 1
    auc = pd.DataFrame({'auc': [(num1+num2*0.5)/RunTime]})
    return auc

if __name__=='__main__':
    g1, user_list, edgedata = Construct_Network()
    Cal_NonExistEdge(g1)
    g2 = Construct_Training(user_list, edgedata)
    origin_norm_user_feature = Normalization_User_Feature(user_list)
    norm_user_feature_30 = Top30_User_Feature(user_list, origin_norm_user_feature)
    for j in range(20):
        revise_user_feature = Revise_User_Feature(user_list, origin_norm_user_feature, norm_user_feature_30,  j, g2)
        exist_data_similarity = Test_ExistData(g2, revise_user_feature)
        noexist_data_similarity = Test_NoExistData(g2, revise_user_feature)
        auc = Cal_AUC(exist_data_similarity, noexist_data_similarity)
        auc.to_csv("./result/origin/%s/%s_auc%s.txt"%(args.network_name, args.network_name, args.iter), header=None, index=False, mode='a')










