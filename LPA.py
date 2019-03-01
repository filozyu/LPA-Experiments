import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn import metrics

# training params
DEFAULT_PARAMS = {
    'NUM_CLASSES': 2,
    'NUM_TRIALS': 10,
    'NUM_TRAIN': range(4, 100, 10)
}
# Currently only support for incremental integer labels (starts from 0),
# nodes without labels should be left with ''

READ_PATH = './sample_graphs/'
WRITE_PATH = './results/'
# TODO: finish {read,write}_path

class LPA(object):
    def __init__(self, graph, params):
        self.graph = graph
        self.params = params

    def read_graph(self, path=None):
        pass
    def write_results(self, path=None):
        pass

    def set_default_params(self):

        if 'NUM_CLASSES' not in self.params.keys():
            raise Exception("Number of classes not specified")
        if 'NUM_TRIALS' not in self.params.keys():
            self.params['NUM_TRIALS'] = 10
        if 'NUM_TRAIN' not in self.params.keys():
            if self.params['NUM_CLASSES'] == 2:
                self.params['NUM_TRAIN'] = range(4, 100, 10)
            elif self.params['NUM_CLASSES'] == 10:
                self.params['NUM_TRAIN'] = range(14, 200, 10)

    def set_test_train(self, nodes, num_train, num_classes):
        """
        Mask the majority of the nodes' labels to create an environment for SSL
        """

        train_ids = []
        nodes["test_train"] = "UNLABELLED"
        all_nodes = list(nodes.index)

        while len(train_ids) < num_classes:
            for i in range(num_classes):
                current_class = nodes.loc[nodes["label"] == i].index
                samp_id = np.random.choice(current_class)
                train_ids.append(samp_id)

        while len(train_ids) < num_train:
            samp_id = np.random.randint(low=0, high=len(all_nodes))
            if samp_id in train_ids:
                continue
            train_ids.append(samp_id)

        if len(train_ids) > num_train:
            raise Exception("Incorrect sample number")

        nodes.loc[train_ids, 'test_train'] = "LABELLED"
        train_nodes = nodes.loc[nodes["test_train"] == "LABELLED"]

        if train_nodes['label'].nunique() != num_classes:
            raise Exception("The sampled set does not contain every class")

        return nodes

    def cal_cmn(self, nodes, train_ids, f_u):
        """
        Calculate class mass normalisation, with add-one smoothing prior
        """
        num_classes = self.params['NUM_CLASSES']
        train_nodes = nodes.loc[train_ids, "label"]
        q_addone = np.zeros(num_classes)
        for i in range(num_classes):
            q_addone[i] = (1 + len(train_nodes.loc[train_nodes == i].index)) / (
                        num_classes + len(train_nodes.index))

        normaliser = np.sum(f_u, axis=0)
        q_temp = np.multiply(q_addone, 1 / normaliser)
        cmn_raw = np.multiply(f_u, q_temp)  # [num_test x num_classes]
        cmn = np.argmax(cmn_raw, axis=1)  # [num_test]

        return cmn

    def onehot_encoding(self, nodes, train_ids):
        """
        Onehot encoding of labelled data, return shape [num_train x num_classes]
        """
        train_nodes = nodes.loc[train_ids, "label"]
        return pd.get_dummies(train_nodes).values


    def lpa(self, demo_run=False):
        """
        Label propagation, write to nodes the threshold prediction and CMN prediction
        """

        if demo_run == False:
            if self.params['NUM_TRIALS'] != 1:
                self.params['NUM_TRIALS'] = 1
            if len(self.params['NUM_TRAIN']) != 1:
                self.params['NUM_TRAIN'] = [0]

        else:
            self.set_default_params()

        num_classes = self.params['NUM_CLASSES']
        nodes = pd.DataFrame()
        nodes['label'] = nx.get_node_attributes(self.graph, 'label').values()

        W = nx.to_numpy_array(self.graph)
        D = np.diag(W.sum(0))

        cmn_result = {}
        thresh_result = {}

        for numTrain in self.params['NUM_TRAIN']:
            cmn_result[numTrain] = []
            thresh_result[numTrain] = []
            for t in range(self.params['NUM_TRIALS']):
                if demo_run == True:
                    nodes = self.set_test_train(nodes, numTrain, num_classes)
                    train_ids = nodes.loc[nodes["test_train"] == "LABELLED"].index
                    test_ids = nodes.loc[nodes["test_train"] == "UNLABELLED"].index
                else:
                    train_ids = nodes.loc[nodes["label"] != ''].index
                    test_ids = nodes.loc[nodes["label"] == ''].index

                W_uu = np.take(W, test_ids, axis=0)
                W_uu = np.take(W_uu, test_ids, axis=1)
                D_uu = np.take(D, test_ids, axis=0)
                D_uu = np.take(D_uu, test_ids, axis=1)
                W_ul = np.take(W, test_ids, axis=0)
                W_ul = np.take(W_ul, train_ids, axis=1)

                f_l = self.onehot_encoding(nodes, train_ids) # [num_train x num_classes]

                d_uu_min_wuu = np.subtract(D_uu, W_uu) # [num_test x num_test]
                inv = np.linalg.solve(d_uu_min_wuu, np.eye(np.size(d_uu_min_wuu, 0))) # [num_test x num_test]

                prod = np.matmul(inv, W_ul)  # [num_test x num_train]

                f_u = np.matmul(prod, f_l)  # [num_test x num_classes]

                thresh = np.argmax(f_u, axis=1)  # [num_test]
                cmn = self.cal_cmn(nodes, train_ids, f_u)  # [num_test]

                if demo_run == True:
                    test_true = nodes.loc[test_ids, "label"].values  # [num_test]
                    acc_thresh = metrics.accuracy_score(y_true=test_true, y_pred=thresh)
                    acc_cmn = metrics.accuracy_score(y_true=test_true, y_pred=cmn)
                    thresh_result[numTrain].append(acc_thresh)
                    cmn_result[numTrain].append(acc_cmn)

        if demo_run is True:
            t_mean_res = []
            c_mean_res = []
            for key in thresh_result.keys():
                thres_mean = np.mean(np.array(thresh_result[key]))
                t_mean_res.append(thres_mean)
            for key in cmn_result.keys():
                cmn_mean = np.mean(np.array(cmn_result[key]))
                c_mean_res.append(cmn_mean)

            return t_mean_res, c_mean_res

        else:
            graph = self.graph

            nodes.loc[train_ids, 'thresh'] = nodes.loc[train_ids, 'label'].values
            nodes.loc[test_ids, 'thresh'] = thresh

            nodes.loc[train_ids, 'cmn'] = nodes.loc[train_ids, 'label'].values
            nodes.loc[test_ids, 'cmn'] = cmn

            nx.set_node_attributes(G=graph, name='thresh', values=dict(nodes['thresh']))
            nx.set_node_attributes(G=graph, name='cmn', values=dict(nodes['cmn']))

            results = {'write_to_nodes': ['thresh', 'cmn']}
            return graph, results

    def demo(self):
        t_mean_res, c_mean_res = self.lpa(demo_run=True)
        return t_mean_res, c_mean_res

    def run(self):
        graph, results = self.lpa()
        return graph, results


if __name__ == '__main__':
    print("Running LPA...")
    in_graph = self.read_graph(path=READ_PATH)
    lpa = LPA(in_graph, DEFAULT_PARAMS)
    out_graph, results = lpa.run()
    self.write_results(path=WRITE_PATH)
    print("Finished!")
