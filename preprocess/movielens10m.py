import pandas as pd
import numpy as np
import time

MOVIELENS_DATA = './datasets/movielens/'

NUMBER_USERS_MOVIELENS = 71567
NUMBER_ITEMS_MOVIELENS = 10681

def preprocess():
    print("STARTING CSV FILE LOAD\n======================")
    start_time = time.time()

    ratings_list = [i.strip().split("::") for i in open(MOVIELENS_DATA + 'ratings.dat', 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(MOVIELENS_DATA + 'movies.dat', 'r').readlines()]

    ratings_df = pd.DataFrame(ratings_list, columns=['userId', 'movieId', 'rating', 'timestamp'], dtype=int)
    movies_df = pd.DataFrame(movies_list, columns=['movieId', 'title', 'genres'])
    movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)
    movies_df.head()
    ratings_df.head()
    R_df = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    R_df.head()
    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    print("Loaded CSV files in %s seconds\n ---" % (time.time() - start_time))

    f = open("./datasets/movielens/movielens.mat", 'w')

    print("STARTING CLUTO FORMAT PROCESSING\n================================")
    start_time = time.time()
    for i in range(R_demeaned.shape[0]):
        for j in range(R_demeaned.shape[1]):
            if R_demeaned[i][j] != 0:
                f.write(str(j) + ' ' + str(R_demeaned[i][j]) + ' ')
        f.write('\n')

    f.close()
    print("Processed data in %s seconds\n ---" % (time.time() - start_time))
    matrixFileName = "./datasets/movielens/movielens.mat"
    return matrixFileName
