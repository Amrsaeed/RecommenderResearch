# Assign gu = 0.5, to every user u.
# Compute the initial clustering of users with CLUTO.
# while number of users who switched clusters > 1% of the total number of users do
#   Estimate S and Spu , ∀pu ∈ {1, . . . , k} with Equation 4.
#   for all user u don
#       for all cluster pu do
#           Compute gu for cluster pu with Equation 5.
#           Compute the training error.
#       end for
#       Assign user u to the cluster pu that has the smallest training error and update gu to the corresponding one
#       for cluster pu .
#   end for
# end while

import os
from subprocess import call
import csv
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import time

CLUTO_SCLUSTER_EXECUTABLE = "./cluto/scluster"
CLUTO_VCLUSTER_EXECUTABLE = "./cluto/vcluster"
MOVIELENS_DATA = './datasets/movielens/'
NUMBER_USERS_MOVIELENS = 138493
NUMBER_ITEMS_MOVIELENS = 27278
K = 2

print("STARTING CSV FILE LOAD\n======================")
start_time = time.time()

ratings_list = [i.strip().split(",") for i in open(MOVIELENS_DATA + 'ratings.csv', 'r').readlines()]
movies_list = [i.strip().split(",") for i in open(MOVIELENS_DATA + 'movies.csv', 'r').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns=['userID', 'movieID', 'rating', 'timestamp'], dtype=int)
movies_df = pd.DataFrame(movies_list, columns=['movieID', 'title', 'genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
movies_df.head()
ratings_df.head()
R_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
R_df.head()
R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

print("Loaded CSV files in %s seconds\n ---" % (time.time() - start_time))

# rows = csvreader[1:, 0].astype(np.float)
# cols = csvreader[1:, 1].astype(np.float)
# data = csvreader[1:, 2].astype(np.float)
#
# print("STARTING SPARSE MATRIX GENERATION\n=================================")
# start_time = time.time()
# sparse_matrix = csr_matrix((data, (rows, cols)), shape=(NUMBER_USERS_MOVIELENS+1, NUMBER_ITEMS_MOVIELENS+1))
# print("Sparse Matrix Generation in %s seconds\n ---" % (time.time() - start_time))
#
f = open("./datasets/movielens/movielens.mat", 'w')
#
# Nonzero = str(sparse_matrix.nnz)
# sparse_matrix = sparse_matrix.toarray()
# f.write(str(sparse_matrix.shape[0] - 1) + ' ' + str(sparse_matrix.shape[1] - 1) + ' ' + Nonzero)
#

print("STARTING CLUTO FORMAT PROCESSING\n================================")
start_time = time.time()
for i in range(R_demeaned.shape[0]):
    for j in range(R_demeaned.shape[1]):
        if R_demeaned[i][j] != 0:
            f.write(str(j) + ' ' + str(R_demeaned[i][j]) + ' ')
    f.write('\n')

f.close()
print("Processed data in %s seconds\n ---" % (time.time() - start_time))

print("STARTING CLUTO CLUSTERING\n=========================")
start_time = time.time()
if CLUTO_VCLUSTER_EXECUTABLE and os.path.isfile(CLUTO_VCLUSTER_EXECUTABLE):
    commandList = list()
    matrixFileName = "./datasets/movielens/movielens.mat"
    commandList.append(CLUTO_VCLUSTER_EXECUTABLE)
    commandList.append(matrixFileName)
    commandList.append(str(K))

    status = call(commandList)

    print(status)
print("Finished Clustering in %s seconds\n ---" % (time.time() - start_time))
