from cfg_extraction_constants import CFG_FILE
from cfg_parsing import read_graph
from collections import Counter
#import networkx as nx
import os
import pandas as pd

projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]
data_directory = "output-data"
file_cfg_data_mask = "functions-cfg-{}.csv"

map_pairs = {}


def analyze_property_two_edges(cfg_filepath):
    graphs = read_graph(cfg_filepath)

    if graphs is not None:
        cfg_dot = graphs[0]
        edges = [(edge.get_source(), edge.get_destination()) for edge in cfg_dot.get_edges()]

        counts = Counter(edges)
        duplicate_edges = [edge for edge in edges if counts[edge] > 1]
        duplicate_edges = list(set(duplicate_edges))
        for edge in duplicate_edges:
            source = edge[0]
            destination = edge[1]
            print(source, destination)
            source_label = cfg_dot.get_node('{}'.format(source))[0].get_label()
            destination_label = cfg_dot.get_node('{}'.format(destination))[0].get_label()
            print(source, source_label)
            print(destination, destination_label)

            source_node_type = source_label[2:-2]
            source_node_type = source_node_type[0:source_node_type.index(",")]

            destination_node_type = destination_label[2:-2]
            destination_node_type = destination_node_type[0:destination_node_type.index(",")]

            if (source_node_type, destination_node_type) not in map_pairs:
                map_pairs[(source_node_type, destination_node_type)] = 0

            map_pairs[(source_node_type, destination_node_type)] += 1
        print("end of cfg")


def analyze_double_edge(project):
    filepath = os.path.join(data_directory, file_cfg_data_mask.format(project))
    df = pd.read_csv(filepath)
    df = df[df[CFG_FILE].notnull()]

    two_edges_nodes = 0
    total_cfgs = 0
    for index, row in df.iterrows():
        cfg_filepath = row[CFG_FILE]

        cfg_filepath_parts = cfg_filepath.split("/")
        cfg_filename = cfg_filepath_parts[-1]

        if project != "httpd":
            repository_directory = cfg_filepath_parts[-2]
            output_commit_index = -3
            base_directory_max_index = -4
        else:
            repository_directory = ""
            output_commit_index = -2
            base_directory_max_index = -3
        output_commit = cfg_filepath_parts[output_commit_index]
        base_directory = "/".join(cfg_filepath_parts[0:base_directory_max_index])

        reduced_cfg_path = os.path.join(
            base_directory, "{}-reduced".format(project), output_commit, repository_directory, cfg_filename)
        #print(reduced_cfg_path)

        graphs = read_graph(reduced_cfg_path, print_filepath=False)

        if graphs is not None:
            cfg_dot = graphs[0]
            #cfg_nx = nx.nx_pydot.from_pydot(cfg_dot)
            #print([node.get_name() for node in cfg_dot.get_nodes()])
            edges = [(edge.get_source(), edge.get_destination()) for edge in cfg_dot.get_edges()]
            #print(edges)
            number_edges = len(edges)
            number_unduplicated_edges = len(set(edges))
            if number_edges != number_unduplicated_edges:
                print("Two Edges: {}".format(reduced_cfg_path))
                two_edges_nodes += 1

                complete_cfg_path = os.path.join(
                    base_directory, project, output_commit, repository_directory, cfg_filename)
                analyze_property_two_edges(complete_cfg_path)
            total_cfgs += 1
    print("total_cfgs: {}, two_edge_nodes: {}".format(total_cfgs, two_edges_nodes))

    for tuple_types in map_pairs.keys():
        print(tuple_types, map_pairs[tuple_types])

    # gecko-dev
    # total_cfgs: 254160, two_edge_nodes: 20098 (7.91%)

    # linux
    # total_cfgs: 104760, two_edge_nodes: 14846 (14.17%)

    # https
    # total_cfgs: 1064, two_edge_nodes: 234


def main():
    for project in projects[3:4]:
        analyze_double_edge(project)


if __name__ == "__main__":
    main()
