import os
import pydot
import re


files_with_problems = []


def adjust_file(filepath):
    changed_rows = []
    # replace all occurrences of the required string
    with open(filepath, "rt") as f:
        for row in f:
            p = re.compile(r'(.*) \[label = \"(.*)\" \]')
            m = p.match(row.strip())
            if m is not None:
                changed_rows.append('{} [label = "{}" ]\n'.format(m[1], m[2].strip().replace('"', '').replace("\n", "")))
            else:
                changed_rows.append(row)

    # overwrite the input file with the resulting data
    fin = open(filepath, "wt")
    fin.write("".join(changed_rows))
    fin.close()

    graphs = pydot.graph_from_dot_file(filepath)
    return graphs


def read_graph(filepath, print_filepath=True):
    graphs = pydot.graph_from_dot_file(filepath)
    if print_filepath:
        print(filepath)
    return graphs


def get_graph_name(cfg_directory, filename):
    filepath = os.path.join(cfg_directory, filename)
    graphs = read_graph(filepath)
    if graphs is None:
        graphs = adjust_file(filepath)
        files_with_problems.append(filename)
    if graphs is None:
        return ""
    graph = graphs[0]
    print(filename, graph.get_name().replace('"', ''))
    return graph.get_name().replace('"', '')


def map_cfg_per_function(cfg_directory):
    map_cfg = {}
    files = os.listdir(cfg_directory)
    files.sort()
    for filename in files:
        if filename.endswith(".dot"):
            function_graph_name = get_graph_name(cfg_directory, filename)
            map_cfg[function_graph_name] = filename
    #print("map_cfg ", map_cfg)
    print("len(map_cfg) ", len(map_cfg.keys()))
    print("files_with_problems ", files_with_problems)
    print("len(files_with_problems) ", len(files_with_problems))
    return map_cfg

