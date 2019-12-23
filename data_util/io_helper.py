import os


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def ensure_dir_for_file(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
