from cfg_parsing import read_graph
from graphviz import Source
import matplotlib.pyplot as plt
import networkx as nx
import os

cfg_directory = "cfg-data"
statements_cfg_folder = "statements-cfg-output"
reduced_cfg_folder = "reduced-cfg-output"


def write_statement_file(statements_cfg_directory, cfg_name, node_statements):
    statements_filename = os.path.join(statements_cfg_directory, "{}.txt".format(cfg_name))
    with open(statements_filename, 'w') as output_file:
        for node in node_statements:
            output_file.write("{} {}\n".format(node, [label[1:-1] for label in node_statements[node]]))


def write_dot_file(reduced_cfg_directory, cfg_name, cfg_nx_merged):
    cfg_filename = os.path.join(reduced_cfg_directory, "{}.dot".format(cfg_name))
    nx.drawing.nx_pydot.write_dot(cfg_nx_merged, cfg_filename)


def reduce_graph(cfg_dot, show_graph=False):
    #print(cfg_dot)
    cfg_nx = nx.nx_pydot.from_pydot(cfg_dot)
    #print(cfg_nx)
    #print("indegree", cfg_nx.in_degree())
    #print("outdegree", cfg_nx.out_degree())

    indegree_1 = [node for (node, indegree) in cfg_nx.in_degree() if indegree == 1]
    outdegree_0_1 = [node for (node, outdegree) in cfg_nx.out_degree() if outdegree <= 1]

    potentially_to_be_removed = [node for node in indegree_1 if node in outdegree_0_1]
    #print("potentially_to_be_removed ({}): {}".format(len(potentially_to_be_removed), potentially_to_be_removed))

    to_be_kept_in_graph = []
    for node in potentially_to_be_removed:
        predecessor = [pred for pred in cfg_nx.nodes if node in cfg_nx[pred]][0]
        #print("node", node, "predecessor", predecessor)
        if cfg_nx.out_degree[predecessor] > 1:
            # node should be kept in the graph
            to_be_kept_in_graph.append(node)
    nodes_to_be_removed = [node for node in potentially_to_be_removed if node not in to_be_kept_in_graph]

    if show_graph:
        s = Source(cfg_dot, filename="output-cfg-images/0-initial-graph.gv", format="pdf")
        s.view()

    for node in nodes_to_be_removed:
        cfg_dot.get_node('"{}"'.format(node))[0].set_color("blue")

    #s = Source(cfg_dot, filename="output-cfg-images/1-nodes-highlighted.gv", format="pdf")
    #s.view()

    all_nodes = cfg_nx.nodes
    all_edges = cfg_nx.edges

    cfg_nx_new = nx.MultiDiGraph()
    for node in all_nodes:
        cfg_nx_new.add_node(node)

    for edge in all_edges:
        cfg_nx_new.add_edge(edge[0], edge[1])

    #cfg_dot_new = nx.nx_pydot.to_pydot(cfg_nx_new)
    #s = Source(cfg_dot_new, filename="output-cfg-images/2-all-nodes.gv", format="pdf")
    #s.view()

    # It will now merge the nodes
    nodes_to_be_added = {node: [node] for node in all_nodes if node not in nodes_to_be_removed}

    nodes_to_be_merged = nodes_to_be_removed
    nodes_to_be_merged.reverse()

    # Adds all the nodes whose predecessor is already part of the graph
    for node in nodes_to_be_merged:
        if cfg_nx.nodes[node] is not None:
            predecessor = [pred for pred in cfg_nx.nodes if node in cfg_nx[pred]][0]
            if predecessor in nodes_to_be_added:
                index_predecessor = nodes_to_be_added[predecessor].index(predecessor)
                nodes_to_be_added[predecessor].insert(index_predecessor + 1, node)
                nodes_to_be_removed.remove(node)

    # All the remaining nodes have a predecessor that was removed from the graph
    nodes_to_be_merged = nodes_to_be_removed
    while len(nodes_to_be_removed) > 0:
        for node in nodes_to_be_removed:
            predecessor = [pred for pred in cfg_nx.nodes if node in cfg_nx[pred]][0]
            for current_node in nodes_to_be_added:
                if predecessor in nodes_to_be_added[current_node]:
                    index_predecessor = nodes_to_be_added[current_node].index(predecessor)
                    nodes_to_be_added[current_node].insert(index_predecessor + 1, node)
                    nodes_to_be_merged.remove(node)

        nodes_to_be_removed = nodes_to_be_merged

    cfg_nx_merged = nx.MultiDiGraph(name=cfg_dot.get_name().replace('"', ''))

    # Adds the nodes for the new merged graph
    node_statements = {}
    for node in nodes_to_be_added:
        node_label = "-".join(nodes_to_be_added[node])
        cfg_nx_merged.add_node(node_label)
        node_statements[node_label] = [cfg_dot.get_node('"{}"'.format(added_node_label))[0].get_label()
                                       for added_node_label in nodes_to_be_added[node]]

    # Prepares the new labels of each node (we need to find the corresponding new node)
    # (This could be optimized somehow)
    new_labels = {}
    for node in all_nodes:
        for new_node in nodes_to_be_added:
            if node in nodes_to_be_added[new_node]:
                new_labels[node] = "-".join(nodes_to_be_added[new_node])

    # Adds the edges for the new merged graph
    for edge in all_edges:
        if new_labels[edge[0]] != new_labels[edge[1]]:
            cfg_nx_merged.add_edge(new_labels[edge[0]], new_labels[edge[1]])

    if show_graph:
        cfg_dot_merged = nx.nx_pydot.to_pydot(cfg_nx_merged)
        s = Source(cfg_dot_merged, filename="output-cfg-images/3-merged-nodes.gv", format="pdf")
        s.view()

    return cfg_nx_merged, node_statements


def check_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    cfg_name = "5-cfg"
    cfg_name = "6-cfg"
    cfg_filename = "{}.dot".format(cfg_name)
    cfg_filepath = os.path.join(cfg_directory, cfg_filename)

    graphs = read_graph(cfg_filepath)
    if graphs is not None:
        cfg_dot = graphs[0]
        cfg_nx_merged, node_statements = reduce_graph(cfg_dot, True)

        check_directory_exists(reduced_cfg_folder)
        check_directory_exists(statements_cfg_folder)

        write_dot_file(reduced_cfg_folder, cfg_name, cfg_nx_merged)
        write_statement_file(statements_cfg_folder, cfg_name, node_statements)


if __name__ == "__main__":
    main()
