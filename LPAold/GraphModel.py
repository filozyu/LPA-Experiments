#!/usr/bin/env python3
# coding: utf-8

from abc import ABCMeta, abstractmethod

import sys
sys.path.append('GCN')
from copy import copy


import networkx as nx
# from model_utils import *



LABEL_NAME = 'label'



class GraphModel(metaclass=ABCMeta):
    """
        Class that implements the Graph Model Abstract Class
    """

    # def __init__(self,A,X,y_train,idx_train,params):
    def __init__(self, graph, params):
        """

        :param graph:  networkx Graph
        :param params: dictionary of parametes for the model / data
        """

        self.graph = graph
        self.params = params
        self.LIST_METRICS = {'roc_auc_score', 'accuracy'}
        #super(GraphModel, self).__init__()

    def initialize_graph(self):
        """

        :return: initialize the default parameters of the model
                and extracts data from the graph
        """
        self.set_default_params(self.params)
        self.extract_data_from_nx_graph()

    def set_default_params(self,DEFAULT_PARAMS={}):
        """
        Set the default parameters in case none is provided
        """

        def cast_like(a, b):
            if isinstance(b, int):
                return int(a)
            elif isinstance(b, float):
                return float(a)
            elif isinstance(b, str):
                return str(a)
            else:
                return a

        # FEATURE indicates whether to use FEATURE or not
        # This also depends on the availability of the features
        for key in DEFAULT_PARAMS.keys():
            if key not in self.params.keys():
                self.params[key] = DEFAULT_PARAMS[key]
            else:
                self.params[key] = cast_like(self.params[key], DEFAULT_PARAMS[key])

    @abstractmethod
    def extract_feature_matrix(self):
        '''

        :param graph: nx graph
        :return: the feature matrix or a identity matrix if no features are involved
        '''
        pass

        # if self.num_features >0:
        #     X = np.zeros([self.num_nodes, self.num_features])
        #
        #     for i, node in enumerate(self.nodes_list):
        #         node_features_dict = copy(self.graph.nodes[node])
        #         node_features_dict.pop('label')
        #         for j, var in enumerate(node_features_dict.keys()):
        #             X[i][j] = node_features_dict[var]
        #     return X
        # else:
        #     X = np.eye(self.num_nodes)
        #     return X



    def extract_data_from_nx_graph(self):
        '''

        :param graph: nx graph
        :return: generates the adjacecy matrix, number of nodes, features, and feature matrix
        '''

        self.A = nx.adjacency_matrix(self.graph)
        self.nodes_list = list(self.graph.nodes)
        self.num_nodes = self.A.shape[0]

        assert self.num_nodes == self.graph.number_of_nodes()

        self.feature_names = list(set(self.graph.nodes[list(self.graph.nodes)[0]].keys()).difference({LABEL_NAME}))


        self.num_feature_names = len(self.feature_names)

        self.X = self.extract_feature_matrix()



    # @abstractmethod
    # def fit_model(self):
    #     pass
    #
    # @abstractmethod
    # def train_model(self):
    #     pass
    #
    # @abstractmethod
    # def get_graph_predictions(self):
    #     pass