from cluto.cluster import cluto_cluster
from preprocess.movielens10m import preprocess as mvp


CLUTO_SCLUSTER_EXECUTABLE = "./cluto/scluster"
CLUTO_VCLUSTER_EXECUTABLE = "./cluto/vcluster"
MOVIELENS_DATA = './datasets/movielens/'
# CLUSTERED_FILE = './sports.mat.clustering.10'

NUMBER_USERS_MOVIELENS = 71567
NUMBER_ITEMS_MOVIELENS = 10681

K = 10

matrixFileName = mvp()
CLUSTERED_FILE = cluto_cluster(K, matrixFileName)

g = [0.5] * NUMBER_USERS_MOVIELENS
g_complement = g
clusters = [int(line.rstrip('\n')) for line in open(CLUSTERED_FILE)]

switched_users = len(g)
switch_percent = 100
print(switched_users)

while switch_percent > 1:
    exit()
    # Estimate S and Spu , ∀pu ∈ {1, . . . , k} with Equation 4.
    #   for all user u do
    #       for all cluster pu do
    #           Compute gu for cluster pu with Equation 5.
    #           Compute the training error.
    #       end for
    #       Assign user u to the cluster pu that has the smallest training error and update gu to the corresponding one for cluster pu .
    #   end for
