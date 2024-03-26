import numpy as np
import pandas as pd
import os
length_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
def precise_and_recall_for_one(user, edge_score, p_r_test_edges, p_r_non_edges):

    score_record = []
    for _ in p_r_test_edges[user]:
        score_record.append([1, edge_score['%s_%s'%(user, _)]])
    for _ in p_r_non_edges[user]:
        score_record.append([0, edge_score['%s_%s'%(user, _)]])
    sorted_score_record = sorted(score_record, key=lambda x: x[1], reverse=True)
    precise_list = []
    for _ in length_list:
        precise_list.append(sum([x[0] for x in sorted_score_record[:_]]) / _)
    recall_list = []
    for _ in length_list:
        recall_list.append(sum([x[0] for x in sorted_score_record[:_]]) / len(p_r_test_edges[user]))
    return precise_list, recall_list

def precise_and_recall(p_r_test_edges, p_r_non_edges, edge_score, tim, data_name, size):
    precise_ = np.array(np.zeros(len(length_list)))
    recall_ = np.array(np.zeros(len(length_list)))
    nodes_list = [x for x in p_r_test_edges.keys()]
    for counter, _ in enumerate(nodes_list):
        p, r = precise_and_recall_for_one(_, edge_score, p_r_test_edges, p_r_non_edges)
        precise_ += np.array(p)
        recall_ += np.array(r)
    precise_ /= len(nodes_list)
    recall_ /= len(nodes_list)
    fout = open(r'../data/baseline/SEAL/pr/%s/size_%s/precise_%s.txt'%(data_name, size, tim), 'w', encoding='utf-8')
    for _ in zip(length_list, precise_):
        fout.write(str(_[0]) + ' ' + str(_[1]) + '\n')
    fout.close()
    fout = open(r'../data/baseline/SEAL/pr/%s/size_%s/recall_%s.txt'%(data_name, size, tim), 'w', encoding='utf-8')
    for _ in zip(length_list, recall_):
        fout.write(str(_[0]) + ' ' + str(_[1]) + '\n')
    fout.close()

def cal_data(pr_data):
    p_r_test_edges = {}
    p_r_non_edges = {}
    edge_score = {}
    for i in range(len(pr_data)):
        if pr_data.iloc[i][3] == 1:
            if pr_data.iloc[i][0] not in p_r_test_edges:
                p_r_test_edges[pr_data.iloc[i][0]] = []
            p_r_test_edges[pr_data.iloc[i][0]].append(pr_data.iloc[i][1])
        else:
            if pr_data.iloc[i][0] not in p_r_non_edges:
                p_r_non_edges[pr_data.iloc[i][0]] = []
            p_r_non_edges[pr_data.iloc[i][0]].append(pr_data.iloc[i][1])
        edge_score['%s_%s' % (pr_data.iloc[i][0], pr_data.iloc[i][1])] = pr_data.iloc[i][2]
    return p_r_test_edges, p_r_non_edges, edge_score

def k_seal():
    # size_list = [100, 500, 1000, 2000, 3000, 4000, 5000]
    size_list = [100, 500, 1000, 2000]
    for data_name in ['twitter']:
        for size in size_list:
            for idx, file in enumerate(os.listdir('../data/baseline/SEAL/pr_origin_data/%s/size_%s' % (data_name, size))):
                pr_data = pd.read_csv('../data/baseline/SEAL/pr_origin_data/%s/size_%s/%s' % (data_name, size, file),
                                      header=None)
                p_r_test_edges, p_r_non_edges, edge_score = cal_data(pr_data)
                precise_and_recall(p_r_test_edges, p_r_non_edges, edge_score, idx+1, data_name, size)

if __name__ == '__main__':
    k_seal()