import os

cfg_dataset_directory = "output-cfg-dataset"


def write_cfgs_to_file(project, dataset_samples):
    cfg_dataset_filepath = check_output_file_exists(project)
    with open(cfg_dataset_filepath, 'a') as output_file:
        for sample in dataset_samples:
            # cfg_filepath,label,size,adjacency_matrix,feature_matrix
            output_file.write("{};{};{};{};{}\n".format(sample[0], sample[1], sample[2], sample[3], sample[4]))
    dataset_samples.clear()


def check_output_file_exists(project):
    if not os.path.exists(cfg_dataset_directory):
        os.makedirs(cfg_dataset_directory)
    cfg_dataset_filepath = os.path.join(cfg_dataset_directory, "cfg-dataset-{}.csv".format(project))
    if not os.path.exists(cfg_dataset_filepath):
        with open(cfg_dataset_filepath, 'w') as output_file:
            output_file.write("cfg_filepath;label;size;adjacency_matrix;feature_matrix\n")
    return cfg_dataset_filepath
