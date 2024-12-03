import ast
from collections import Counter
import os
import networkx as nx
import numpy as np
import pandas as pd

from cfg_extraction_constants import CFG_FILE, LABEL
from cfg_parsing import read_graph
from feature_identifier_constants import *
from save_cfg_features import write_cfgs_to_file


# This is another interesting link to explain GCN (Graph Convolution Network):
# https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780


# The node types can be of the following types (according to Yan2019):
# 1) Code Sequence:
# -- # Numeric Constants: (assignments in our case)
# -- # Transfer Instructions:
# -- # Call Instructions: METHOD
# -- # Arithmetic Instructions: <operator>.sizeOf, <operator>.cast, <operator>.subtraction, <operator>.division
#                               <operator>.preDecrement, <operator>.minus, <operator>.addition, <operator>.preIncrement
# -- # Compare Instructions: <operator>.equals, <operator>.lessThan, <operator>.logicalNot, <operator>.conditional,
#                            <operator>.logicalOr, <operator>.logicalAnd, <operator>.greaterThan, <operator>.notEquals
# -- # Move Instructions: (not considered)
# -- # Termination Instructions: (return, exit), METHOD_RETURN, RETURN
# -- # Data Declaration Instructions: (variable declaration), <operator>.new, <operator>.assignment
# -- # Total Instructions: (not sure what would be the difference between it and the instructions in the vertex)
# 2) Vertex Structure:
# -- # Offspring (Degree)
# -- # Instructions in the Vertex

# <operator>.indirection, <operator>.addressOf
# NS_LITERAL_CSTRING, NS_SUCCEEDED, NS_ENSURE_TRUE, NS_LITERAL_STRING, NS_ASSERTION, NS_FAILED, NS_ADDREF, UNKNOWN
# pushBuffer.Append, getter_AddRefs, buffer.Append, getter_Copies

# All items have 5 or more occurrences, or they are obviously a reserved node (total of 141 node types)

statements_cfg_folder = "statements-cfg-output"
reduced_cfg_folder = "reduced-cfg-output"

data_directory = "output-data"
file_cfg_data_mask = "functions-cfg-{}.csv"
projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]

statement_types = []

other_cases = []
allocation_features = []
deallocation_features = []
unsafe_features = []


def obtain_cfg_data_structures(cfg_filepath, statements_filepath=None):
    A, X, cfg_nx = None, None, None
    if os.path.exists(cfg_filepath):
        graphs = read_graph(cfg_filepath, print_filepath=False)

        if graphs is not None:
            cfg_dot = graphs[0]
            cfg_nx = nx.nx_pydot.from_pydot(cfg_dot)
            # s = Source(cfg_dot, filename="test.gv", format="pdf")
            # s.view()

            # TODO what if we do a topological sorting?
            if len(cfg_nx.nodes()) > 0:
                A, node_order = convert_graph_to_adjacency_matrix(cfg_dot)
                X = obtain_attribute_matrix(cfg_nx, node_order, statements_filepath)
    return A, X, cfg_nx


def convert_graph_to_adjacency_matrix(cfg_dot):
    cfg = nx.nx_pydot.from_pydot(cfg_dot)
    #node_types = obtain_node_types(cfg_dot)

    #print(node_types)

    #adjacency_matrix = nx.adjacency_matrix(cfg)
    #print(type(adjacency_matrix))
    #print(adjacency_matrix)

    #np_matrix = nx.to_numpy_matrix(cfg, nodelist=cfg.nodes())
    #print(np_matrix)
    #print(cfg.nodes())

    different_order_nodes = list(cfg.nodes())
    if len(different_order_nodes) > 1:
        head = different_order_nodes.pop(-2)
        different_order_nodes.insert(0, head)
    #print(different_order_nodes)
    np_matrix = nx.to_numpy_matrix(cfg, nodelist=different_order_nodes)
    #print(np_matrix)
    return np_matrix, different_order_nodes


def obtain_statements(statements_filepath):
    with open(statements_filepath) as f:
        node_statements = f.readlines()

    statement_count = {}
    statements_node = {}
    for i in range(len(node_statements)):
        node = node_statements[i]
        space_index = node.index(" ")
        node_label = node[0:space_index]
        str_statements = node[space_index + 1:].replace("\n", "")

        # to read the string list, we need to use "ast"
        # str_list  = ['(RETURN,return rv;,return rv;)']
        statements = ast.literal_eval(str_statements)
        statement_type = [tp[1:tp.index(",")] for tp in statements]
        statement_types.extend(statement_type)

        statement_count[node_label] = len(statements)
        statements_node[node_label] = statements
    return statement_count, statements_node


def get_node_type_id(statement_type):
    if statement_type in statement_type_map:
        return statement_type_map[statement_type]
    else:
        return -1


def obtain_code_sequence_features(number_features_code_sequence, statements):
    # creates the features based on the code sequence
    X_code_sequence = np.zeros(number_features_code_sequence)
    types = []
    for statement in statements:
        statement_type = statement[1:statement.index(",")]
        types.append(statement_type)
        node_type_id = get_node_type_id(statement_type)
        if node_type_id != -1:
            X_code_sequence[node_type_id] += 1
    #print(types)
    return X_code_sequence


def count_functions_statement(statement, functions, features_identified):
    count_functions = 0
    for function_name in functions:
        if function_name in statement:
            if statement.startswith("{}(".format(function_name)) or " {}(".format(function_name) in statement:
                count_functions += 1
                print(statement)
                features_identified.append((function_name, statement))
            else:
                print("DIFFERENT CASE (function_name, statement): ({}, {})".format(function_name, statement))
                other_cases.append((function_name, statement))
    return count_functions


def count_allocation_functions(statement):
    allocation_functions_list = ["malloc", "calloc", "realloc", "new",
                                 "kmalloc", "vmalloc", "kvmalloc", "vzalloc",
                                 "kcalloc"]
    return count_functions_statement(statement, allocation_functions_list, allocation_features)


def count_deallocation_functions(statement):
    deallocation_functions_list = ["free", "delete", "vfree", "kfree", "kvfree"]
    return count_functions_statement(statement, deallocation_functions_list, deallocation_features)


def count_convert_unsafe_functions(statement):
    convert_unsafe_functions_list = ["atoi", "atol", "atoll", "atof"]
    return count_functions_statement(statement, convert_unsafe_functions_list, unsafe_features)


def count_string_unsafe_functions(statement):
    string_unsafe_functions_list = ["gets", "getpw", "strcat", "strcpy", "sprintf", "vsprintf", "puts", "read_chunk"]
    return count_functions_statement(statement, string_unsafe_functions_list, unsafe_features)


def count_scanf_unsafe_functions(statement):
    scanf_unsafe_functions_list = ["scanf", "fscanf", "sscanf", "vscanf", "vsscanf", "vfscanf"]
    return count_functions_statement(statement, scanf_unsafe_functions_list, unsafe_features)


def count_other_unsafe_functions(statement):
    scanf_other_functions_list = ["realpath", "getopt", "getpass", "streadd", "strecpy", "strtrns"]
    return count_functions_statement(statement, scanf_other_functions_list, unsafe_features)


def count_address_of(statement_type):
    return 1 if statement_type == "<operator>.addressOf" else 0
    # Verificar este caso
    # ---> Nem todos são deste tipo.
    # Ver este caso: /hdd/josep/linux/output-cfg-aa77d26961fa4ecb11fe4209578dcd62ad15819d/security---selinux---hooks.c/233-cfg.dot


def count_pointer_assignment(statement_type, statement):
    if statement_type == "<operator>.assignment" and statement.startswith("*") and ("=" in statement):
        print(statement_type, statement)
        return 1
    return 0


def obtain_feature_mm_count(cfg_statement):
    # TODO parses the statement and check all the types
    list_features = {}
    statement_type = cfg_statement[1:cfg_statement.index(",")]
    statement = cfg_statement[cfg_statement.index(",")+1:-1]
    list_features[ALLOCATION_FUNCTIONS] = count_allocation_functions(statement)
    list_features[DEALLOCATION_FUNCTIONS] = count_deallocation_functions(statement)
    list_features[POINTER_ASSIGNMENT] = count_pointer_assignment(statement_type, statement)
    list_features[MEMORY_ADDRESS_OF] = count_address_of(statement_type)
    list_features[CONVERT_UNSAFE] = count_convert_unsafe_functions(statement)
    list_features[STRING_UNSAFE] = count_string_unsafe_functions(statement)
    list_features[SCANF_UNSAFE] = count_scanf_unsafe_functions(statement)
    list_features[OTHER_UNSAFE] = count_other_unsafe_functions(statement)
    return list_features


def obtain_mm_features(number_features_memory_management, statements):
    # Method to calculate the features related to lack of memory management
    X_memory_management = np.zeros(number_features_memory_management)
    for statement in statements:
        #print(statement)
        mm_features = obtain_feature_mm_count(statement)
        for mm_feature in mm_features:
            X_memory_management[mm_feature] += mm_features[mm_feature]
    return X_memory_management


def obtain_attribute_matrix(cfg_nx, node_order, statements_filepath):
    # This is a matrix with node degrees. Other attributes can be used
    # TODO Normalize this matrix

    number_features_code_sequence = 8
    number_features_vertex_structure = 3
    number_features_memory_management = 8
    total_features = number_features_vertex_structure + \
                     number_features_code_sequence + \
                     number_features_memory_management

    cfg_order = len(node_order)  # number of nodes in the CFG
    if statements_filepath is None:
        statements_count, statements_node = {node_label: 1 for node_label in node_order}, None
    else:
        statements_count, statements_node = obtain_statements(statements_filepath)

    X = np.zeros((cfg_order, total_features))
    for i in range(X.shape[0]):
        X[i][0] = cfg_nx.out_degree(node_order[i])  # outdegree
        X[i][1] = cfg_nx.in_degree(node_order[i])  # indegree
        X[i][2] = statements_count[node_order[i]]  # number of statements
        X[i][number_features_vertex_structure:number_features_vertex_structure+number_features_code_sequence] = \
            obtain_code_sequence_features(number_features_code_sequence, statements_node[node_order[i]])
        X[i][number_features_vertex_structure + number_features_code_sequence:total_features] = \
            obtain_mm_features(number_features_memory_management, statements_node[node_order[i]])

    if statements_filepath is None:
        # removes the column with the number of statements per node
        X = np.delete(X, 2, 1)
    #print(X)
    return X


def get_node_type(label):
    # remove the quotes and parenthesis
    node_type = label[2:-2]
    node_type = node_type[0:node_type.index(",")]
    return node_type


def obtain_node_types(cfg_dot):
    node_types = {}
    for node in cfg_dot.get_nodes():
        node_label = node.get_label()
        #print(node.get_name(), node_label)
        if node_label is not None:
            node_type = get_node_type(node_label)
            if node_type not in node_types:
                node_types[node_type] = 1
            else:
                node_types[node_type] += 1
    #print(node_types)
    return node_types


def obtain_reduced_statement_filepath(project, cfg_filepath):
    cfg_filepath_parts = cfg_filepath.split("/")

    cfg_filename = cfg_filepath_parts[-1]
    cfg_name = cfg_filename[:cfg_filename.index(".dot")]
    if project not in ["httpd"]:
        repository_directory = cfg_filepath_parts[-2]
        output_commit_index = -3
        base_directory_max_index = -4
    else:
        repository_directory = ""
        output_commit_index = -2
        base_directory_max_index = -3

    output_commit = cfg_filepath_parts[output_commit_index]
    base_directory = "/".join(cfg_filepath_parts[0:base_directory_max_index])

    cfg_reduced_filepath = os.path.join(base_directory, "{}-reduced".format(project),
                                        output_commit, repository_directory, "{}.dot".format(cfg_name))
    statements_filepath = os.path.join(base_directory, "{}-statements".format(project),
                                       output_commit, repository_directory, "{}.txt".format(cfg_name))
    return cfg_reduced_filepath, statements_filepath


def read_cfg_file(project):
    filepath = os.path.join(data_directory, file_cfg_data_mask.format(project))
    df = pd.read_csv(filepath, delimiter=";")
    df = df[df[CFG_FILE].notnull()]

    dataset_samples = []
    for index, row in df.iterrows():
        cfg_filepath = row[CFG_FILE]
        cfg_reduced_filepath, statements_filepath = obtain_reduced_statement_filepath(project, cfg_filepath)

        A, X, cfg_nx = obtain_cfg_data_structures(cfg_reduced_filepath, statements_filepath)

        if A is not None:
            adjacency_matrix = list(A.getA1().flatten())
            adjacency_matrix = [int(a) for a in adjacency_matrix]
            features = list(X.flatten())
            features = [int(f) for f in features]
            dataset_samples.append((cfg_reduced_filepath, row[LABEL], A.shape[0],
                                    adjacency_matrix, features))

        if (index + 1) % 10 == 0:
            write_cfgs_to_file(project, dataset_samples)

    write_cfgs_to_file(project, dataset_samples)


def write_output_file(rows, output_filepath):
    rows_list = list(set(rows))
    with open(output_filepath, 'w') as output_file:
        for item in rows_list:
            output_file.write("{},{}\n".format(item[0], item[1]))


def main():
    # This is just a sample for the feature extraction of the reduced CFG already
    cfg_name = "6-cfg"
    cfg_filepath = os.path.join(reduced_cfg_folder, "{}.dot".format(cfg_name))
    statements_filepath = os.path.join(statements_cfg_folder, "{}.txt".format(cfg_name))

    A, X, cfg_nx = obtain_cfg_data_structures(cfg_filepath, statements_filepath)
    #print(A)
    #print(X)

    for project in projects[3:4]:
        read_cfg_file(project)
        write_output_file(other_cases, "other-cases.csv")
        write_output_file(allocation_features, "allocation.csv")
        write_output_file(deallocation_features, "deallocation.csv")
        write_output_file(unsafe_features, "unsafe.csv")

        #statement_type_count = Counter(statement_types)
        #for item in statement_type_count.most_common():
        #    print(item)


if __name__ == "__main__":
    main()
