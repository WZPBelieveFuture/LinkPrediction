import networkx as nx
from Node2vertor import Node2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

RunNumber = 1
Number = 11016
user_data = pd.read_csv('../data/origin_data/twitter/userList.txt', header=None)
user_list = list(user_data.iloc[:][0])
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


def train_model(g, train_exist, test_exist, train_noexist, test_noexist):
    train_data = pd.concat([train_exist, train_noexist], ignore_index=True)
    train_label = np.zeros(len(train_data), dtype=np.int)
    train_label[:len(train_exist)] = 1
    test_data = pd.concat([test_exist, test_noexist], ignore_index=True)
    test_label = np.zeros(len(test_data), dtype=np.int)
    test_label[:len(test_exist)] = 1
    node_vec = GetEmbedding(g)
    train_feature = Cal_feature(node_vec, train_data, 'l2')
    clf = LogisticRegression(n_jobs=-1).fit(train_feature, train_label)
    return clf, node_vec

def get_prob(model, user1, user2, node2vec):
    feature = Get_feature(node2vec, user1, user2, 'l2')
    return model.predict_proba(feature)[0][1]

def precise_and_recall_for_one(model, user, node2vec):

    score_record = []
    for _ in p_r_test_edges[user]:
        score_record.append([1, get_prob(model, user, _, node2vec)])
    for _ in p_r_non_edges[user]:
        score_record.append([0, get_prob(model, user, _, node2vec)])

    sorted_score_record = sorted(score_record, key=lambda x: x[1], reverse=True)

    precise_list = []
    for _ in length_list:
        precise_list.append(sum([x[0] for x in sorted_score_record[:_]]) / _)
    recall_list = []
    for _ in length_list:
        recall_list.append(sum([x[0] for x in sorted_score_record[:_]]) / len(p_r_test_edges[user]))
    return precise_list, recall_list

def precise_and_recall(model, g, test_exist, node_vec, tim):
    get_testing_edges_for_p_r(test_exist)
    get_non_edges_for_p_r(g)
    precise_ = np.array(np.zeros(len(length_list)))
    recall_ = np.array(np.zeros(len(length_list)))
    nodes_list = [x for x in p_r_test_edges.keys()]
    for counter, _ in enumerate(nodes_list):
        p, r = precise_and_recall_for_one(model, _, node_vec)
        precise_ += np.array(p)
        recall_ += np.array(r)
    precise_ /= len(nodes_list)
    recall_ /= len(nodes_list)
    fout = open(r'../result/twitter/precise_%s.txt'%tim, 'w', encoding='utf-8')
    for _ in zip(length_list, precise_):
        fout.write(str(_[0]) + ' ' + str(_[1]) + '\n')
    fout.close()
    fout = open(r'../result/twitter/recall_%s.txt'%tim, 'w', encoding='utf-8')
    for _ in zip(length_list, recall_):
        fout.write(str(_[0]) + ' ' + str(_[1]) + '\n')
    fout.close()

def average_feature(u, v):
    return [sum(item) / 2.0 for item in zip(u, v)]

def hammord_feature(u, v):
    return [x * y for x, y in zip(u, v)]

def l1_distance_feature(u, v):
    return [abs(x - y) for x, y in zip(u, v)]

def l2_distance_feature(u, v):
    return [(x - y) ** 2 for x, y in zip(u, v)]

def Get_feature(node_vertor, user1, user2, prop):
    features = []
    x = node_vertor[user1]
    y = node_vertor[user2]
    feature = []
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

def Cal_feature(node_vertor, data, prop):
    features = []
    for i in range(len(data)):
        feature = []
        x = node_vertor[data.iloc[i][0]]
        y = node_vertor[data.iloc[i][1]]
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
    for tim in range(20):
        g, train_exist, test_exist, train_noexist, test_noexist = spilt_train_test()
        model, node_vec = train_model(g, train_exist, test_exist, train_noexist, test_noexist)
        precise_and_recall(model, g, test_exist, node_vec, tim+1)
