import os

import numpy as np


def prepare_dirs(root_dir: str):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)


def read_pandas(corpus_filepath, cols=None):
    import pandas as pd
    res = pd.read_csv(corpus_filepath, compression='gzip', usecols=cols)

    return res


def load_corpus(input_df, col_name='content_preprocessed'):
    corpus_texts = input_df[col_name].values
    return corpus_texts


def train_embeds(corpus_texts, embedder, sentence_embedding_path):
    if os.path.exists(sentence_embedding_path):
        print('corpus loading from %s' % sentence_embedding_path)
        passage_embeddings = np.load(sentence_embedding_path)
    else:
        print('num rows %d', len(corpus_texts))
        passage_embeddings = embedder.encode(
            corpus_texts, show_progress_bar=True)
        passage_embeddings = np.array(
            [embedding for embedding in passage_embeddings]).astype("float32")
        with open(sentence_embedding_path, 'wb') as f:
            np.save(f, passage_embeddings)
        print('corpus saved to %s' % sentence_embedding_path)
    print('Num embeddings %d' % passage_embeddings.shape[0])


def get_pytorch_model(root_dir, model_name='all-mpnet-base-v2'):
    """
    model = get_pytorch_model(root_data_dir)
    """
    from sentence_transformers import SentenceTransformer

    models_dir = os.path.join(root_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        print('huggingface model loading...')
        embedder = SentenceTransformer(model_name)
        embedder.save(model_path)
    else:
        print('pretrained model loading...')
        embedder = SentenceTransformer(model_name_or_path=model_path)
    print('model loadind done')

    return embedder
