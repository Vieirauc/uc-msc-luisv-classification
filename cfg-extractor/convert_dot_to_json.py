# dot_to_json_graph.py
# http://stackoverflow.com/questions/40262441/how-to-transform-a-dot-graph-to-json-graph

# Packages needed  :
# sudo aptitude install python-networkx python-pygraphviz python-pydot
#
# Syntax :
# python dot_to_json_graph.py graph.dot

from cfg_parsing import adjust_file, get_graph_name
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import pandas as pd

base_dot_directory = "/opt/josep"
commit_data_directory = "function-data"
output_directory = "output-data"
projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]
commit_data_mask = "{}-functions.csv"
VULNERABLE_COMMIT_HASH = "Vulnerable Commit Hash"
FILE_PATH = "File Path"
VULNERABLE_FUNCTIONS = "Vulnerable File Functions"

graph_type = "cfg"


def convert_dot_to_json(project):
    project_dir = os.path.join(base_dot_directory, project)
    wrong_files = []
    for directory in os.listdir(project_dir)[0:1]:
        print(directory)
        files = os.listdir(os.path.join(project_dir, directory))
        for filename in files:
            try:
                dot_filepath = os.path.join(project_dir, directory, filename)
                adjust_file(dot_filepath)
                dot_graph = nx.nx_pydot.read_dot(dot_filepath)
                print(json.dumps(json_graph.node_link_data(dot_graph)))
            except TypeError:
                wrong_files.append(dot_filepath)
    print("Files with problems ({}): {} ".format(len(wrong_files), wrong_files))


def directory_per_file(project):
    return project in ["glibc", "gecko-dev", "linux"]


def write_output_file(output_filename, rows):
    header = 'commit;filepath;function_name;CFG_filepath;vulnerable_label\n'
    output_filepath = os.path.join(output_directory, output_filename)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    with open(output_filepath, 'w') as output_file:
        output_file.write(header)
        for row in rows:
            output_file.write(row)


def map_functions_to_cfg(project):
    # Goals of this function:
    #  1) reads the DF of functions per project
    #  2) iterates over the functions of a file in a commit
    #  3) finds the corresponding CFG file (.dot) of the function
    #  4) creates a CSV file with the output, with the following fields
    #     commit, filepath, function_name, CFG_filepath, vulnerable_label
    filepath = os.path.join(commit_data_directory, commit_data_mask.format(project))
    df = pd.read_csv(filepath)

    # Filter the samples without vulnerable functions
    df = df[df[VULNERABLE_FUNCTIONS].notnull()]

    csv_rows = []
    for index, row in df.iterrows():
        commit = row[VULNERABLE_COMMIT_HASH]
        filepath = row[FILE_PATH]

        cfg_output_directory = os.path.join(base_dot_directory, project, "output-{}-{}".format(graph_type, commit))
        if directory_per_file(project):
            filepath_directory = filepath.replace("/", "---")
            cfg_output_directory = os.path.join(cfg_output_directory, filepath_directory)

        map_function_name_cfgfile = {}
        for cfg_file in os.listdir(cfg_output_directory):
            function_name_dot = get_graph_name(cfg_output_directory, cfg_file)
            map_function_name_cfgfile[function_name_dot] = os.path.join(cfg_output_directory, cfg_file)

        functions = json.loads(row[VULNERABLE_FUNCTIONS])
        for function in functions:
            function_name = function["Name"]
            vulnerable = function["Vulnerable"] == "Yes"

            if function_name in map_function_name_cfgfile:
                cfg_filepath = map_function_name_cfgfile[function_name]
            else:
                # some cases are not found due to the "scope resolution operator" (::)
                # for instance, "nsIndexedToHTML::AsyncConvertData", but in the CSV file
                # only "AsyncConvertData" is present.
                # So we need to find it regardless the scope operator.
                function_in_map = ["::{}".format(function_name) in k for k in map_function_name_cfgfile.keys()]

                if True in function_in_map:
                    key_index = function_in_map.index(True)
                    complete_function_name = list(map_function_name_cfgfile.keys())[key_index]
                    cfg_filepath = map_function_name_cfgfile[complete_function_name]
                else:
                    cfg_filepath = ""
                    print("Houston, we have a problem...")

            csv_row = "{};{};{};{};{}\n".format(commit, filepath, function_name, cfg_filepath, vulnerable)
            print(csv_row)
            csv_rows.append(csv_row)

    write_output_file("functions-cfg-{}.csv".format(project), csv_rows)


def main():
    for project in projects[0:1]:
        #if directory_per_file(project):
        #    map_functions_to_cfg(project)
        #else:
        #    convert_dot_to_json(project)
        map_functions_to_cfg(project)


if __name__ == "__main__":
    main()
