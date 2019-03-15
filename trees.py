import numpy as np
import networkx as nx
import time


def rst_spine(num_rst,graph,epsilon=0.2,weight=False):
    rst_list = []
    sp_list = []
    while len(set(rst_list)) < num_rst:
        print("Generating RST No.",len(set(rst_list))+1,"...")
        start = time.time()
        visited_nodes = []
        visited_edges = []
        rst_root = np.random.choice(graph.nodes)
        curr_node = rst_root
        visited_nodes.append(rst_root)
        while len(set(visited_nodes)) < graph.number_of_nodes():
            neighb_list = [node for node in graph.neighbors(curr_node)]

            if weight:
                if len(set(visited_nodes)) > 0.9 * graph.number_of_nodes():
                    epsilon = 0.8
                weight_list = [graph.get_edge_data(curr_node,node)['weight'] for node in graph.neighbors(curr_node)]
                trans_prob = weight_list/sum(weight_list)
                if np.random.random() > epsilon:
                    next_node = np.random.choice(a=neighb_list,p=trans_prob)
                else:
                    next_node = np.random.choice(a=neighb_list)

            else:
                next_node = np.random.choice(a=neighb_list)

            if next_node not in visited_nodes:
                visited_edges.append((curr_node,next_node))
            visited_nodes.append(next_node)
            curr_node = next_node
            
        rst = nx.Graph(visited_edges)
        print("Tree? :",nx.is_tree(tree))
        spine = [node for node in nx.dfs_preorder_nodes(rst,rst_root)]
        rst_list.append(rst)
        sp_list.append(spine)
        end = time.time()
        print("Time:",end-start,"\n")

    print("Finished!")
    return rst_list,sp_list
