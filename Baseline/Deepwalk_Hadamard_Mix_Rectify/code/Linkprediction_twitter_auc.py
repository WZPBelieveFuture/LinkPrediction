import networkx as nx
from Node2vertor import Node2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import copy

RunNumber = 1
Number = 11016
user_data = pd.read_csv('../data/origin_data/twitter/userList.txt', header=None)
user_list = list(user_data.iloc[:][0])

def GetEmbedding(G):
    node_vertor = {}
    model = Node2Vec(G, walk_length=60, num_walks=10, p=1, q=1, workers=1)
    model.train(window_size=8)
    embeddings = model.get_embeddings(user_list, Number)
    for i in user_list:
        node_vertor[i] = embeddings[i]
    return node_vertor

def spilt_train_test():
    follow_data = pd.read_csv('../data/origin_data/twitter/followgraph.txt', header=None)
    G_origin = nx.DiGraph()
    G_origin.add_nodes_from(user_list)
    G_origin.add_edges_from(zip(follow_data.iloc[:][0], follow_data.iloc[:][1]))
    no_exist_edge = np.array(list(nx.non_edges(G_origin)))
    _, test_noexist = train_test_split(no_exist_edge, test_size=0.001)
    _, train_noexist = train_test_split(no_exist_edge, test_size=0.009)
    G = nx.DiGraph()
    G.add_nodes_from(user_list)
    train_exist, test_exist = train_test_split(follow_data, test_size=0.1)
    G.add_edges_from(zip(train_exist.iloc[:][0], train_exist.iloc[:][1]))
    return G, train_exist, test_exist, pd.DataFrame(train_noexist), pd.DataFrame(test_noexist)

def Normalization_User_Feature():
    user_norm_feature_dict = {}
    data=pd.read_csv('../data/origin_data/twitter/userfeature.txt', sep='\t', header=None)
    for i in range(len(data)):
        user_feature = np.array(data.loc[i][0:100])
        user_norm_feature_dict[user_list[i]] = user_feature/np.sum(user_feature)
    return user_norm_feature_dict

def Top30_User_Feature(feature_dict):
    user_norm_feature_dict_30 = {}
    for i in user_list:
        user_feature = np.zeros(100, dtype=np.float)
        user_feature[np.argpartition(feature_dict[i], -30)[-30:]] = feature_dict[i][np.argpartition(feature_dict[i], -30)[-30:]]
        user_norm_feature_dict_30[i] = user_feature/np.sum(user_feature)
    return user_norm_feature_dict_30

def Revise_User_Feature(origin_feature, top30_feature, k, g):
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

def Cal_Auc(g, user_topic_feature, train_exist, test_exist, train_noexist, test_noexist):
    train_data = pd.concat([train_exist, train_noexist], ignore_index=True)
    train_label = np.zeros(len(train_data), dtype=np.int)
    train_label[:len(train_exist)] = 1
    test_data = pd.concat([test_exist, test_noexist], ignore_index=True)
    test_label = np.zeros(len(test_data), dtype=np.int)
    test_label[:len(test_exist)] = 1
    node_vec = GetEmbedding(g)
    train_feature = Cal_feature(node_vec, user_topic_feature, train_data, 'ham')
    clf = LogisticRegression(n_jobs=-1).fit(train_feature, train_label)
    test_feature = Cal_feature(node_vec, user_topic_feature, test_data, 'ham')
    proba = clf.predict_proba(test_feature)
    auc_roc = roc_auc_score(test_label, [y for x, y in proba])
    pd.DataFrame({'auc': [auc_roc]}).to_csv('../result/twitter/auc.csv', header=None, index=None, mode='a')

def average_feature(u, v):
    return [sum(item) / 2.0 for item in zip(u, v)]


def hammord_feature(u, v):
    return [x * y for x, y in zip(u, v)]


def l1_distance_feature(u, v):
    return [abs(x - y) for x, y in zip(u, v)]


def l2_distance_feature(u, v):
    return [(x - y) ** 2 for x, y in zip(u, v)]


def Cal_feature(node_vertor, user_topic_feature, data, prop):
    features = []
    for i in range(len(data)):
        feature = []
        x = node_vertor[data.iloc[i][0]]
        x = np.concatenate([x, user_topic_feature[data.iloc[i][0]]])
        y = node_vertor[data.iloc[i][1]]
        y = np.concatenate([y, user_topic_feature[data.iloc[i][1]]])
        if prop is 'all' or prop is 'avg':
            feature = average_feature(x, y)
        if prop is 'all' or prop is 'ham':
            feature = hammord_feature(x, y)
        if prop is 'all' or prop is 'l1':
            feature = l1_distance_feature(x, y)
        if prop is 'all' or prop is 'l2':
            feature = l2_distance_feature(x, y)
        features.append(feature)
        
    return np.array(features)

if __name__ == '__main__':
    for _ in range(20):
        g, train_exist, test_exist, train_noexist, test_noexist = spilt_train_test()
        origin_norm_user_feature = Normalization_User_Feature()
        norm_user_feature_30 = Top30_User_Feature(origin_norm_user_feature)
        revise_user_feature = Revise_User_Feature(origin_norm_user_feature, norm_user_feature_30,  1, g)
        Cal_Auc(g, revise_user_feature, train_exist, test_exist, train_noexist, test_noexist)