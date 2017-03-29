import copy
import csv
import pickle
import random

import numpy as np
import time

data_mat_const = pickle.load(open('Hospital_dataset.pickle', 'rb')).astype(np.int64)
ground_truth = np.asarray(list(csv.reader(open('Complications - Hospital - Encoded.csv'))), dtype=int)

numS, numE, numA = data_mat_const.shape[0], data_mat_const.shape[1], data_mat_const.shape[2]

max_it = 20


def evaluate(truth_val, total_claims):
    return (np.count_nonzero(truth_val[:, :, [8]] - ground_truth_new[:, :, [8]]) + np.count_nonzero(
        truth_val[:, :, [9]] - ground_truth_new[:, :, [9]])) / (total_claims * 2)
    # return np.count_nonzero(truth_val - ground_truth) / (numE * numA)


if __name__ == '__main__':
    # Parameters
    start = time.clock()
    truth = np.zeros(shape=(numS, numE, numA), dtype=np.int64)
    ground_truth_new = np.zeros(shape=(numS, numE, numA), dtype=np.int64)
    num_Claims = np.zeros(shape=(numS,), dtype=np.int64)
    claim_confident = [[1.0 for a in range(numA)] for e in range(numE)]
    cleaned_set = []
    num_answered = 0  # Number of claimed entities
    num_errors = 0
    total_claims = 0

    knowledge_pairs = [(8, 9)]
    # knowledge_pairs = [(0, 1)]

    # Voting Initialization
    for e in range(numE):
        if np.max(data_mat_const[:, e]) > 0:
            num_answered += 1

    for s in range(numS):
        for e in range(numE):
            if not np.all(data_mat_const[s][e] == -1):
                num_Claims[s] += 1
    for s in range(numS):
        for e in range(numE):
            for a in range(numA):
                if np.max(data_mat_const[s, e]) > 0:
                    ground_truth_new[s][e][a] = ground_truth[e][a]

    # Count num_Claims
    for s in range(numS):
        for e in range(numE):
            if not np.all(data_mat_const[s][e] == -1):
                for a in range(8, 10):
                    if data_mat_const[s][e][a] != ground_truth_new[s][e][a]:
                        num_errors += 1
                total_claims += 1

    # Evaluate the result
    #print('Error rate: {0}'.format(evaluate(truth, total_claims)))
    print('Data Error rate: {0}'.format(num_errors / (total_claims * 2)))

    num_changed = 0
    num_change2_true = 0
    whole_set = []
    for a in range(30000, 60000):
        whole_set.append(a)

    for pair in knowledge_pairs:
        print(pair)
        att_claim_list = data_mat_const[:, :, [pair[0], pair[1]]].tolist()
        unique_0 = np.unique(data_mat_const[:, :, pair[0]], return_index=False)
        unique_0 = unique_0.tolist()
        determine_set = []
        remained_set = copy.copy(unique_0)
        fact_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
        samples = np.random.uniform(0, 1, numE * numA)
        i = 0
        for e in range(numE):

            for s in range(numS):

                if np.max(data_mat_const[s, e]) > 0:
                    #print(s, e, pair[1])
                    #print(fact_list)
                    if samples[i] < 0.5:  # Change dependence attribute
                        if att_claim_list[s][e][0] in fact_list:
                            # print(truth[s][e][pair[0]])
                            # print(truth[s][e][pair[1]])

                            # print(truth[s][e][0])

                            # print(fact_list[truth[s][e][pair[0]]])
                            if fact_list[att_claim_list[s][e][0]] != att_claim_list[s][e][1]:
                                # num_changed += 1
                                # if fact_list[att_claim_list[s][e][0]] == ground_truth[e][pair[1]]:
                                #     num_change2_true += 1
                                att_claim_list[s][e][1] = fact_list[att_claim_list[s][e][0]]
                                num_changed += 1
                                if att_claim_list[s][e][1] == ground_truth_new[s][e][pair[1]]:
                                    num_change2_true += 1
                        else:
                            fact_list[att_claim_list[s][e][0]] = att_claim_list[s][e][1]
                            # determine_set.append(att_claim_list[s][e][0])
                            # remained_set.remove(att_claim_list[s][e][0])
                    else:  # Change determine attribute
                        if att_claim_list[s][e][0] in fact_list:
                            if fact_list[att_claim_list[s][e][0]] != att_claim_list[s][e][1]:
                                att_claim_list[s][e][0] = random.choice(whole_set)
                                num_changed += 1
                                if att_claim_list[s][e][0] == ground_truth_new[s][e][pair[0]]:
                                    num_change2_true += 1
                        else:
                            fact_list[att_claim_list[s][e][0]] = att_claim_list[s][e][1]

                            # fact_list[truth[e][pair[0]]] = truth[e][pair[1]]
            i += 1

    for e in range(numE):
        if np.max(data_mat_const[:, e]) > 0:
            for a in range(8, 10):
                for s in range(numS):
                    if np.max(data_mat_const[s, e]) > 0:
                        truth[s][e][a] = att_claim_list[s][e][a - 8]

    end = time.clock()
    print(end - start)
    print('Precision: {0}'.format(num_change2_true / num_changed))
    print('Recall: {0}'.format(num_change2_true / num_errors))

    # Evaluate the result
    print('Error rate: {0}'.format(evaluate(truth, total_claims)))
