import os

def prepare_dirs(root_dir: str):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
