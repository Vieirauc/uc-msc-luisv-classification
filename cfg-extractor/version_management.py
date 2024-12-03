import git
import json
import os
import pandas as pd

data_directory = "function-data"


def load_commit(repository_directory, commit):
    running_directory = os.getcwd()
    os.chdir(repository_directory)

    print("Repository Directory: {}".format(repository_directory))
    repo = git.Repo(repository_directory)
    current_commit = repo.head.commit
    print("Current commit: {}\n".format(current_commit))

    checkout_command = "git checkout {}".format(commit)
    my_cmd = os.popen(checkout_command).read()
    print(my_cmd)
    current_commit = repo.head.commit

    print("Repository changed head to commit: {}\n".format(repo.head.commit))

    os.chdir(running_directory)
    return current_commit

