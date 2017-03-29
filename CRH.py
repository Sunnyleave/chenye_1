import csv
import pickle
import numpy as np
import time
data_mat_const = pickle.load(open('Hospital_dataset.pickle', 'rb')).astype(np.int64)
ground_truth = np.asarray(list(csv.reader(open('Complications - Hospital - Encoded.csv'))), dtype=int)

numS, numE, numA = data_mat_const.shape[0], data_mat_const.shape[1], data_mat_const.shape[2]

max_it = 5


def evaluate(truth_val, n_answered):
    return np.count_nonzero(truth_val - ground_truth) / (n_answered * numA)
    # return np.count_nonzero(truth_val - ground_truth) / (numE * numA)


if __name__ == '__main__':
    # Parameters
    start = time.clock()
    w = np.ones(shape=(numS,))
    num_Claims = np.ones(shape=(numS,), dtype=np.int64)
    truth = np.zeros(shape=(numE, numA), dtype=np.int64)
    claim_confident = [[1.0 for a in range(numA)] for e in range(numE)]
    num_answered = 0
    num_errors = 0
    total_claims = 0

    # Count num_Claims
    for s in range(numS):
        for e in range(numE):
            if not np.all(data_mat_const[s][e] == -1):
                num_Claims[s] += 1

    # knowledge_pairs = [(0, 1), (8, 9)]
    #knowledge_pairs = [(0, 1)]

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
                for a in range(numA):
                    if data_mat_const[s][e][a] != ground_truth[e][a]:
                        num_errors += 1
                    total_claims += 1

    # Evaluate the result
    print('Error rate: {0}'.format(evaluate(truth, num_answered)))
    print('Data Error rate: {0}'.format(num_errors / total_claims))

    # Iteratively solve the problem
    for it in range(max_it):
        print('Iteration: {0}'.format(it + 1))
        data_mat = np.copy(data_mat_const)
        num_changed = 0
        num_change2_true = 0

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

        # Update Truth
        for e in range(numE):
            for a in range(numA):
                claim_list = data_mat[:, e, a]
                claim_list_raw = data_mat_const[:, e, a]
                claim_species = np.unique(claim_list, return_index=False)
                wk = np.zeros(shape=(claim_species.shape[0],))
                for k in range(len(claim_species)):
                    wk[k] = np.sum((claim_list == claim_species[k]).astype(int) * w)  # Change here if partial coverage
                claim_confident[e][a] = [claim_species, wk]
                # Select most confident claim
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
        end = time.clock()
        print(end - start)
        print('Precision: {0}'.format(num_change2_true / num_changed))
        print('Recall: {0}'.format(num_change2_true / num_errors))

        # Evaluate the result
        print('Error rate: {0}'.format(evaluate(truth, num_answered)))
        print(w)
