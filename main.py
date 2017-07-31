from cluto.cluster import cluto_cluster
from preprocess.movielens10m import preprocess as mvp
import csv
import numpy as np

CLUTO_SCLUSTER_EXECUTABLE = "./cluto/scluster"
CLUTO_VCLUSTER_EXECUTABLE = "./cluto/vcluster"
MOVIELENS_DATA = './datasets/movielens/'

NUMBER_USERS_MOVIELENS = 71567
NUMBER_ITEMS_MOVIELENS = 10681

K = 10

matrixFileName = mvp()
CLUSTERED_FILE = cluto_cluster(K, matrixFileName)

clusters = [int(line.rstrip('\n')) for line in open(CLUSTERED_FILE)]
rows_count = 0
columns_count = 0

with open("movielensCluto.mat", 'r') as f:
    first_line = f.readline()
    rows_count, columns_count, _ = first_line.split()

R_training = np.zeros((K, int(rows_count), int(columns_count)))

with open("R.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for i in range(int(rows_count)):
        R_training[clusters[i]][i] = next(csvreader)

g = [0.5] * int(rows_count)
g_complement = g
switched_users = len(g)
switch_percent = 100

print(switched_users)

while switch_percent > 1:
    exit()
    # Estimate S and Spu , u  {1, . . . , k} with Equation 4.
    #   for all user u do
    #       for all cluster pu do
    #           Compute gu for cluster pu with Equation 5.
    #           Compute the training error.
    #       end for
    #       Assign user u to the cluster pu that has the smallest training error and update gu to the corresponding one for cluster pu .
    #   end for
