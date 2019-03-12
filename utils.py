import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io as sio

def load_data():
    graph_dict = {}
    graph_keys = ['baseballhockey','pcmac','windowsmac']
    for key in graph_keys:
        weightMat = sio.loadmat(file_name="./sample_graphs/20newsgroups/"+key+"/"+key+"_graph.mat")['W']
        labelMat = sio.loadmat(file_name="./sample_graphs/20newsgroups/"+key+"/"+key+"_Y.mat")['Y']
        labelDict = dict(zip(np.arange(labelMat.shape[0]), labelMat[:,0]))
        graph_dict[key]= nx.Graph(weightMat)
        nx.set_node_attributes(G=graph_dict[key], name='label', values=labelDict)
    return graph_dict

def save_data(graphs):
    return

def plot_result(thresh_dict,cmn_dict):
    y = range(4, 100, 10)
    for key in results_dict.key():
        plt.plot(y,thresh_dict[key],'bv-')
        plt.plot(y,cmn_dict[key],'rv-')
        plt.xlim(0,100)
        plt.ylim(0.0, 1.0)
        plt.xlabel('Number of labelled points')
        plt.ylabel('Mean classification accuracy')
        plt.legend(('Threshold','CMN'))
        plt.title('Baseball vs Hockey')
