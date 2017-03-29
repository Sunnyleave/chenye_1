import csv
import json
import os
import pickle
import random

import numpy as np


def gen_mistakes(truth, a, data_dict):
    claim_range = len(data_dict[a])
    while True:
        wrong = int(np.random.uniform(0, claim_range - 1))
        if truth != wrong:
            return wrong


def sim_sources(data_list, data_dict):
    mu, thresh = 0, 2
    sigma_vec = [0.1, 0.1, 0.1, 0.1, 1]
    numS, numE, numA = len(sigma_vec), len(data_list), len(data_list[0])
    claim_mat = np.ones(shape=(numS, numE, numA)) * -1
    num_wrong = 0

    for s in range(numS):
        # Generate categorical claims.
        for e in range(numE):
            x = random.random()
            if x > 0.5:
                for a in range(numA):
                    rand_seed = abs(np.random.normal(mu, sigma_vec[s]))
                    if abs(rand_seed) < thresh:

                        claim_mat[s, e, a] = data_list[e, a]
                    elif abs(rand_seed) > thresh and a <= 7:
                        claim_mat[s, e, a] = data_list[e, a]
                    else:
                        num_wrong += 1
                        claim_mat[s, e, a] = gen_mistakes(data_list[e, a], a, data_dict)

    print('Error rate of synthetic data: {0}'.format(num_wrong / (numS * numE * numA)))

    return claim_mat


def gen_dictionary(data_list):
    rtn_list = [{} for x in range(len(data_list[0]))]
    data_mat = np.asarray(data_list)

    for i in range(data_mat.shape[1]):
        seq = 0
        col_list = data_mat[:, i].tolist()
        for j in range(data_mat.shape[0]):
            if col_list[j] not in rtn_list[i]:
                rtn_list[i][col_list[j]] = seq
                seq += 1

    return rtn_list


def encoding(data_list, data_dict):
    for i in range(len(data_list)):
        for j in range(len(data_list[1])):
            data_list[i][j] = data_dict[j][data_list[i][j]]

    return data_list


if __name__ == '__main__':

    if not os.path.exists('Complications - Hospital - Encoded.csv'):
        with open('Complications_52k - Hospital.csv', 'r') as fp:
            hospital_list = list(csv.reader(fp))
            del (hospital_list[0])

        '''
        [
        'Provider ID', 'Hospital Name', 'Address', 'City', 'State', 'ZIP Code', 'County Name', 'Phone Number', 'Measure Name',
        'Measure ID', 'Compared to National', 'Denominator', 'Score', 'Lower Estimate', 'Higher Estimate', 'Footnote',
        'Measure Start Date', 'Measure End Date'
        ]
        '''
        hospital_list = np.asarray(hospital_list, dtype=str)[:, 0:10].tolist()

        '''Encode claims'''
        hospital_dict = gen_dictionary(hospital_list)

        '''Replace claims with codes'''
        hospital_list = np.asarray(encoding(hospital_list, hospital_dict), dtype=int)

        np.savetxt('Complications - Hospital - Encoded.csv', hospital_list, fmt='%d', delimiter=',')

        json.dump(hospital_dict, open('Complications - Hospital - Dict.json', 'w'))

    hospital_list = np.asarray(list(csv.reader(open('Complications - Hospital - Encoded.csv'))), dtype=int)

    hospital_dict = json.load(open('Complications - Hospital - Dict.json'))

    claims = sim_sources(hospital_list, hospital_dict)

    pickle.dump(claims, open('Hospital_dataset.pickle', 'wb'))
