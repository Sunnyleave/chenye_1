import csv
import pickle

import numpy as np
import time

data_mat_const = pickle.load(open('Hospital_dataset.pickle', 'rb')).astype(np.int64)
ground_truth = np.asarray(list(csv.reader(open('Complications - Hospital - Encoded.csv'))), dtype=int)

numS, numE, numA = data_mat_const.shape[0], data_mat_const.shape[1], data_mat_const.shape[2]

max_it = 5


def unique_pair(att_list):
    tmp_list = {}
    rtn_list = []
    score_set = []
    for i in range(len(att_list)):
        for j in range(len(att_list[0])):
            key = str(att_list[i][j][0]) + ',' + str(att_list[i][j][1])
            if key not in tmp_list:
                tmp_list[key] = np.zeros(shape=(numS,), dtype=np.int64)
            tmp_list[key][i] = 1
    for key in tmp_list:
        if '-1' not in key:
            fields = [int(x) for x in key.split(',')]
            rtn_list.append([fields[0], fields[1], tmp_list[key]])
            score_set.append([fields[0], fields[1], 0])
    return rtn_list, score_set


def fd_clean_right(data_matrix, fd_results, knowledge_pair):
    for e in range(numE):
        wrong_set = np.zeros(shape=(numS,), dtype=np.int64)
        for s_1 in range(numS):
            if max(data_matrix[s_1][e]) > 0:
                first_elem = data_matrix[s_1][e][knowledge_pair[0]]
                second_elem = data_matrix[s_1][e][knowledge_pair[1]]
                # fd_result = fd_results[first_elem]
                fd_result = fd_results[fd_results[:, 0] == first_elem][0]
                # Judge if current element fit the restriction
                if first_elem == int(fd_result[0]) and second_elem == int(fd_result[1]):
                    continue
                else:
                    wrong_set[s_1] = 1
                    if len(fd_results[fd_results[:, 0] == first_elem].flatten()) > 0:
                        data_matrix[s_1][e][knowledge_pair[1]] = fd_results[fd_results[:, 0] == first_elem].flatten()[1]

    return data_matrix


def fd_clean_both(data_matrix, fd_results, knowledge_pair):
    for e in range(numE):
        wrong_set = np.zeros(shape=(numS,), dtype=np.int64)

        for s_1 in range(numS):
            if max(data_matrix[s_1][e]) > 0:
                first_elem = data_matrix[s_1][e][knowledge_pair[0]]
                second_elem = data_matrix[s_1][e][knowledge_pair[1]]
                # fd_result = fd_results[first_elem]
                fd_result = fd_results[fd_results[:, 0] == first_elem][0]
                # Judge if current element fit the restriction
                if first_elem == int(fd_result[0]) and second_elem == int(fd_result[1]):
                    continue
                else:
                    wrong_set[s_1] = 1
                    score_first, score_second = 0, 0
                    for s_2 in range(numS):
                        if first_elem == data_matrix[s_2][e][knowledge_pair[0]]:
                            score_first += w[s_2]
                        elif second_elem == data_matrix[s_2][e][knowledge_pair[1]]:
                            score_second += w[s_2]
                    if len(fd_results[fd_results[:, 0] == first_elem].flatten()) > 0 and len(
                            fd_results[fd_results[:, 1] == second_elem].flatten()) > 0:
                        if score_first >= score_second:
                            # Change the second to constrained value
                            data_matrix[s_1][e][knowledge_pair[1]] = \
                            fd_results[fd_results[:, 0] == first_elem].flatten()[1]
                        else:
                            # Change the first to constrained value
                            data_matrix[s_1][e][knowledge_pair[0]] = \
                            fd_results[fd_results[:, 1] == second_elem].flatten()[0]
                    elif len(fd_results[fd_results[:, 0] == first_elem].flatten()) > 0:
                        data_matrix[s_1][e][knowledge_pair[0]] = fd_results[fd_results[:, 0] == first_elem].flatten()[1]
                    elif len(fd_results[fd_results[:, 1] == second_elem].flatten()) > 0:
                        data_matrix[s_1][e][knowledge_pair[1]] = fd_results[fd_results[:, 1] == second_elem].flatten()[
                            0]

    return data_matrix


def evaluate(truth_val, n_answered):
    return np.count_nonzero(truth_val - ground_truth) / (n_answered * 2)
# return np.count_nonzero(truth_val - ground_truth) / (numE * numA)
#def evaluate(truth_val, t_claims):
    #return (np.count_nonzero(truth_val[:, :, 8] - ground_truth_new[:, :, 8]) + np.count_nonzero(
        #truth_val[:, :, 9] - ground_truth_new[:, :, 9])) / t_claims


if __name__ == '__main__':
    # Parameters
    start = time.clock()
    w = np.ones(shape=(numS,))
    num_Claims = np.zeros(shape=(numS,), dtype=np.int64)
    truth = np.zeros(shape=(numE, numA), dtype=np.int64)
    claim_confident = [[1.0 for a in range(numA)] for e in range(numE)]
    cleaned_set = []
    num_answered = 0  # Number of claimed entities
    num_errors = 0
    total_claims = 0
    print(numE)
    knowledge_pairs = [(8, 9)]

    # Count num_Claims
    for s in range(numS):
        for e in range(numE):
            if not np.all(data_mat_const[s][e] == -1):
                num_Claims[s] += 1

    # Voting Initialization
    for e in range(numE):
        if np.max(data_mat_const[:, e]) > 0:
            for a in range(numA):
                claim_list = data_mat_const[:, e, a].tolist()
                claim_list = np.asarray([x for x in claim_list if x != -1])
                truth[e][a] = np.argmax(np.bincount(claim_list))
            num_answered += 1

    # Count num_Claims
    for s in range(numS):
        for e in range(numE):
            if not np.all(data_mat_const[s][e] == -1):
                for a in range(8, 10):
                    if data_mat_const[s][e][a] != ground_truth[e][a]:
                        num_errors += 1
                    total_claims += 1

    # Evaluate the result
    print('Error rate: {0}'.format(evaluate(truth, num_answered)))
    print('Data Error rate: {0}'.format(num_errors / total_claims))

    # Iteratively solve the problem
    for it in range(max_it):
        print('Iteration: {0}'.format(it + 1))

        # Copy const matrix
        data_mat = np.copy(data_mat_const)
        num_changed = 0
        num_change2_true = 0

        # Clean the claims
        for pair in knowledge_pairs:
            att_claim_list = data_mat[:, :, [pair[0], pair[1]]].tolist()
            claim_species, score_list = unique_pair(att_claim_list)
            for k in range(len(claim_species)):
                score_list[k][2] = np.sum(claim_species[k][2] * w)  # Change here if partial coverage
            score_list = np.asarray(score_list)
            score_list = score_list[score_list[:, 0].argsort()]
            # Find pair confident
            f_set = np.unique(score_list[:, 0]).astype(np.int64).tolist()
            fd_guide = []
            # fd_guide = np.zeros(shape=(len(f_set), 3))
            for item in f_set:
                tmp_set = score_list[score_list[:, 0] == item]
                fd_guide.append(tmp_set[np.argmax(tmp_set[:, 2]), :])
                # fd_guide[item, :] = tmp_set[np.argmax(tmp_set[:, 2]), :]
            fd_guide = np.asarray(fd_guide)

            # Clean data according to FD
            if (pair[0] not in cleaned_set) and (pair[1] not in cleaned_set):
                data_mat = fd_clean_both(data_mat, fd_guide, pair)
                cleaned_set.append(pair[0])
                cleaned_set.append(pair[1])
            else:
                data_mat = fd_clean_right(data_mat, fd_guide, pair)
                cleaned_set.append(pair[1])

        # Update Truth
        for e in range(numE):
            if np.max(data_mat_const[:, e]) > 0:
                for a in range(numA):
                    claim_list = data_mat[:, e, a]
                    claim_list_raw = data_mat_const[:, e, a]
                    claim_species = np.unique(claim_list, return_index=False)
                    wk = np.zeros(shape=(claim_species.shape[0],))
                    for k in range(len(claim_species)):
                        wk[k] = np.sum(
                            (claim_list == claim_species[k]).astype(int) * w)  # Change here if partial coverage
                    claim_confident[e][a] = [claim_species, wk]
                    most_confident_claim = 0
                    most_confident_confidence = -1
                    for ii in range(len(claim_confident[e][a][0])):
                        if claim_confident[e][a][0][ii] != -1:
                            if most_confident_confidence < claim_confident[e][a][1][ii]:
                                most_confident_confidence = claim_confident[e][a][1][ii]
                                most_confident_claim = claim_confident[e][a][0][ii]
                    truth[e, a] = most_confident_claim

                    for ii in range(len(claim_list_raw)):
                        if claim_list_raw[ii] != -1 and claim_list_raw[ii] != most_confident_claim:
                            num_changed += 1
                            if most_confident_claim == ground_truth[e, a]:
                                num_change2_true += 1

        # truth_new = np.zeros(shape=(numS, numE, numA), dtype=np.int64)
        # for e in range(numE):
        #     if np.max(data_mat_const[:, e]) > 0:
        #         for a in range(8, 10):
        #             for s in range(numS):
        #                 if np.max(data_mat_const[s, e]) > 0:
        #                     truth_new[s][e][a] = truth[e, a]
        # ground_truth_new = np.zeros(shape=(numS, numE, numA), dtype=np.int64)
        # for s in range(numS):
        #     for e in range(numE):
        #         for a in range(numA):
        #             if np.max(data_mat_const[s, e]) > 0:
        #                 ground_truth_new[s][e][a] = ground_truth[e][a]

        print('Precision: {0}'.format(num_change2_true / num_changed))
        print('Recall: {0}'.format(num_change2_true / num_errors))

        # Update Weight
        w_old = w
        score1 = np.zeros(shape=(numS,))
        # Calculate all the costs
        for e in range(numE):
            for a in range(numA):
                for s in range(numS):
                    score1[s] += int((truth[e, a] != data_mat[s, e, a]))
                    # print(truth[e, a], data_mat[s, e, a], int((truth[e, a] != data_mat[s, e, a])))
        score1 = score1 / num_Claims
        w = -np.log(score1 / max(score1) + 1e-300) + 0.00001
        end = time.clock()
        print(end - start)
        # Evaluate the result
        print('Error rate: {0}'.format(evaluate(truth, num_answered)))
        print(w)
