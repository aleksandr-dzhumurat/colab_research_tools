import os


def prepare_dirs(root_dir: str):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)


def read_pandas(corpus_filepath, cols=None):
    import pandas as pd
    res = pd.read_csv(corpus_filepath, compression='gzip', usecols=cols)

    return res


