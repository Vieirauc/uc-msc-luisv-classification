import os
import subprocess


def obtain_directories(repository_path, run_per_directory):
    if run_per_directory:
        return [directory
                for directory in os.listdir(repository_path)
                if os.path.isdir(os.path.join(repository_path, directory)) and
                not directory.startswith(".")]
    return [""]


def extract_cfg(base_output_directory, project, repository_path, commit, run_per_directory=False):
    # joern-parse  --language c [repository_path]
    # joern-export --repr cfg --out [output_directory]

    graph_type = "cfg"
    output_directory = os.path.join(base_output_directory, project,
                                    "output-{}-{}".format(graph_type, commit))

    # If the directory exists, the parsing was already performed for that commit
    if not os.path.exists(output_directory):
        # Command to parse the source code
        for directory in obtain_directories(repository_path, run_per_directory):
            full_directory = os.path.join(repository_path, directory)
            parse_command = "joern-parse  --language c {}".format(full_directory)
            print(subprocess.Popen(parse_command, shell=True, stdout=subprocess.PIPE).stdout.read())

            # Command to export the cfg
            cfg_output_directory = os.path.join(output_directory, directory)
            parse_command = "joern-export --repr {} --out {}".format(graph_type, cfg_output_directory)
            print(subprocess.Popen(parse_command, shell=True, stdout=subprocess.PIPE).stdout.read())

    return output_directory


def extract_cfg_per_file(base_output_directory, project, repository_path, commit, files):
    # joern-parse [repository_path]
    # joern-export --repr cfg --out [output_directory]

    graph_type = "cfg"
    output_directory = os.path.join(base_output_directory, project,
                                    "output-{}-{}".format(graph_type, commit))

    # If the directory exists, the parsing was already performed for that commit
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        for filename in files:
            filepath = os.path.join(repository_path, filename)
            parse_command = "joern-parse {}".format(filepath)
            print(subprocess.Popen(parse_command, shell=True, stdout=subprocess.PIPE).stdout.read())

            # Command to export the cfg
            filepath_directory = filename.replace("/", "---")
            cfg_output_directory = os.path.join(output_directory, filepath_directory)
            parse_command = "joern-export --repr {} --out {}".format(graph_type, cfg_output_directory)
            print(subprocess.Popen(parse_command, shell=True, stdout=subprocess.PIPE).stdout.read())

    return output_directory
