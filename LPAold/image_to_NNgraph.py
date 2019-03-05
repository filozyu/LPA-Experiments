import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

class image_to_NNgraph(object):
    '''
    Convert images to graphs based on nearest neighbours
    image a is connected to image b iff a is one of b's k-NN OR b is one of a's k-NN
    
    Args:
        paths: list (each entry is the path to a txt or CSV file of a set of images 
               that will be used to build the nearest neighbour graph)
               e.g. ["./my_directory/data1.txt","./my_directory/data2.txt"]
                    
        radius: double (this will be used in the weight function)
        
        num_neighbours: int (number of nearest neighbours to build the graph)
    '''
    def __init__(self, paths, num_neighbours, radius=380):
        self.r = radius
        self.k = num_neighbours
        self.data = []
        for path in paths:
            file = pd.read_csv(filepath_or_buffer=path, sep=' ', header=None).dropna(axis=1).values
            self.data.append(file)
            
    def weight(self,x,y):
        return np.exp(-np.sum((x-y)**2)/(self.r**2))


    def CompleteGraph(self):
        result = []
        t = time.time()
        for X in self.data:
            weight_mat = np.zeros((X.shape[0],X.shape[0]))
            for i in range(X.shape[0]):
                for j in range(i+1, X.shape[0]):
                    weight_mat[i,j] = self.weight(X[i,:],X[j,:])
            W = weight_mat + weight_mat.T
            sparse_graph = sparse.csc_matrix(W)
            nx_graph = nx.from_scipy_sparse_matrix(sparse_graph, edge_attribute='weight')
            print("This graph is connected: ",nx.is_connected(nx_graph))
            result.append(nx_graph)
        print("Run time: ",time.time()-t)
        return result
    

    def NNGraph(self,opt=1):
        result = []
        t = time.time()
        
        # OPTION 1
        #############################
        if opt == 1:
            for X in self.data:
                neigh = NearestNeighbors(n_neighbors=self.k)
                neigh.fit(X)
                neigh_index = neigh.kneighbors(X=X,n_neighbors=self.k+1,return_distance=False)
                weight_mat = np.zeros((X.shape[0],X.shape[0]))
                for i in range(1,self.k+1):
                    for (r,c) in zip(neigh_index[:,0],neigh_index[:,i]):
                        weight_mat[r,c] = self.weight(X[r,:],X[c,:])

                for i in range(X.shape[0]):
                    ind = np.where(weight_mat[i,:]!=0)[0]
                    for j in ind:
                        if weight_mat[j,i] == 0:
                            weight_mat[j,i] = weight_mat[i,j]

                sparse_graph = sparse.csc_matrix(weight_mat)
                nx_graph = nx.from_scipy_sparse_matrix(sparse_graph, edge_attribute='weight')
                print("This graph is connected: ",nx.is_connected(nx_graph))
                result.append(nx_graph)
            print("Run time: ",time.time()-t)
        #############################
        
        # OPTION 2
        #############################
        if opt == 2:
            for X in self.data:
                neigh = NearestNeighbors(n_neighbors=self.k)
                neigh.fit(X)
                graph = neigh.kneighbors_graph(X=X, n_neighbors=self.k+1, mode='connectivity')
                graph = graph - sparse.identity(X.shape[0])
                graph = graph.tolil()
                for (i,j) in zip(graph.nonzero()[0],graph.nonzero()[1]):
                    graph[i,j] = self.weight(X[i,:],X[j,:])
                    if graph[j,i] == 0:
                        graph[j,i] = graph[i,j]

                nx_graph = nx.from_scipy_sparse_matrix(graph, edge_attribute='weight')
                print("This graph is connected: ",nx.is_connected(nx_graph))
                result.append(nx_graph)
            print("Run time: ",time.time()-t)
        #############################
        
        return result
    
    def save_edge_list(self,saving_paths,method="N"):
        '''
            Args:
                saving_paths: list (length must be the same as paths in the __init__)      
                e.g. ["./my_directory/data1_kNN.csv","./my_directory/data2_kNN.csv"]
                
                method: "C" for complete graph, "N" for nearest neighbour graph (default is "N")
        '''
        if method == "N":
            result = self.NNGraph()
        elif method == "C":
            result = self.CompleteGraph()
        else:
            raise Exception("invalid method, method has to be either 'N' or 'C'")

        for (i,j) in enumerate(saving_paths):
            nx.write_weighted_edgelist(result[i], path=j, delimiter=',')
            
    
            
            
        