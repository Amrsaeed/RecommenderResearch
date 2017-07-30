import time
from subprocess import call
import os

CLUTO_SCLUSTER_EXECUTABLE = "./scluster"
CLUTO_VCLUSTER_EXECUTABLE = "./vcluster"


def cluto_cluster(K, matrixFileName):
    print("STARTING CLUTO CLUSTERING\n=========================")
    start_time = time.time()
    if CLUTO_VCLUSTER_EXECUTABLE and os.path.isfile(CLUTO_VCLUSTER_EXECUTABLE):
        commandList = list()
        commandList.append(CLUTO_VCLUSTER_EXECUTABLE)
        commandList.append(matrixFileName)
        commandList.append(str(K))

        status = call(commandList)

        print(status)
    print("Finished Clustering in %s seconds\n ---" % (time.time() - start_time))