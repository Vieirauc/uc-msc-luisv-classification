from cfg_extraction import extract_cfg, extract_cfg_per_file
from cfg_parsing import map_cfg_per_function
from cfg_persistence import save_cfg
import os
import pandas as pd
from version_management import load_commit

target_commit = "3b365793c19aff95d1cf9bbea19f138752264d12"

base_project_directory = "/home/zema-21/Projects/"
commit_data_directory = "function-data"
commit_data_mask = "{}-functions.csv"
VULNERABLE_COMMIT_HASH = "Vulnerable Commit Hash"
FILE_PATH = "File Path"

base_output_directory = "/opt/josep"


def obtain_commits(project):
    # each file of "commit_data_directory" (function-data) is in the GitHub from João Henggeler
    # scripts/output/final/[PROJECT_NAME]/affected-files-[PROJECT_ID]-[PROJECT]-master-branch-[date].csv
    # e.g., "scripts/output/final/mozilla/affected-files-1-mozilla-master-branch-20210401212440.csv"
    filepath = os.path.join(commit_data_directory, commit_data_mask.format(project))
    df = pd.read_csv(filepath)
    commits = df[VULNERABLE_COMMIT_HASH].tolist()
    commits = list(set(commits))
    return commits


def obtain_commits_files(project):
    # each file of "commit_data_directory" (function-data) is in the GitHub from João Henggeler
    # scripts/output/final/[PROJECT_NAME]/affected-files-[PROJECT_ID]-[PROJECT]-master-branch-[date].csv
    # e.g., "scripts/output/final/mozilla/affected-files-1-mozilla-master-branch-20210401212440.csv"
    filepath = os.path.join(commit_data_directory, commit_data_mask.format(project))
    df = pd.read_csv(filepath)
    commits = df[VULNERABLE_COMMIT_HASH].tolist()
    commits = list(set(commits))
    commits_files = {}
    for commit in commits:
        commits_files[commit] = df[df[VULNERABLE_COMMIT_HASH] == commit][FILE_PATH].tolist()
    return commits_files


def check_output_directory(base_output_directory, project):
    output_directory = os.path.join(base_output_directory, project)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)


def should_run_per_directory(project):
    return project in ["glibc", "gecko-dev", "linux"]


def extract_cfg_per_commit(project, commits):
    repository_path = os.path.join(base_project_directory, project)
    check_output_directory(base_output_directory, project)
    for commit in commits:
        load_commit(repository_path, commit)
        cfg_directory = extract_cfg(base_output_directory, project,
                                    repository_path, commit, should_run_per_directory(project))
        map_cfg_per_function(cfg_directory)
        save_cfg()


def extract_cfg_per_commit_file(project, commits_files):
    repository_path = os.path.join(base_project_directory, project)
    check_output_directory(base_output_directory, project)
    for commit in commits_files:
        load_commit(repository_path, commit)
        cfg_directory = extract_cfg_per_file(base_output_directory, project,
                                             repository_path, commit, commits_files[commit])
        #map_cfg_per_function(cfg_directory)
        #save_cfg()


def main():
    projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]
    for project in projects[1:2]:
        if should_run_per_directory(project):
            # Extract the CFG for the listed files only
            commits_files = obtain_commits_files(project)
            extract_cfg_per_commit_file(project, commits_files)
        else:
            # Extract the CFG for all the files of the commit
            commits = obtain_commits(project)
            extract_cfg_per_commit(project, commits)


if __name__ == "__main__":
    main()
