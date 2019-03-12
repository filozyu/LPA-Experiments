import numpy as np
import pandas as pd
import networkx as nx
import time
from LPA import *
from utils import *

def GLPA(in_graphs):
    print("Running LPA...")
    thres = {}
    cmn = {}
    for key in graphs.keys():
        start = time.time()
        print(key)
        thres[key],cmn[key] = LPA(graphs[key], DEFAULT_PARAMS).demo()
        end = time.time()
        print('Time:',end-start)
    print("Finished!")
    return thres, cmn

if __name__ == '__main__':
    in_graphs = load_data()
    thres, cmn = GLPA(in_graphs)
