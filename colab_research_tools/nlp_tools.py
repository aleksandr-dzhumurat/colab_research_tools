import os
import shutil
from collections import Counter

import numpy as np

from . import utils


def prepare_nltk(root_data_dir='/srv/data'):
    nltk_data_dir = os.path.join(root_data_dir, 'nltk_data')
    utils.prepare_dirs(nltk_data_dir)
    import nltk

    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)  # Lemmas
    nltk.download('omw-1.4', download_dir=nltk_data_dir)  # Lemmas
    nltk.download(
        'averaged_perceptron_tagger',
        download_dir=nltk_data_dir)  # POS tags
    # тут почему-то корневую надо указывать ¯\_(ツ)_/¯
    nltk.data.path.append(nltk_data_dir)


def prepare_fasttext(root_data_dir, fasttext_model_name='cc.en.300.bin'):
    import fasttext.util
    import fasttext

    fasttext_model_path = os.path.join(root_data_dir, fasttext_model_name)
    if not os.path.exists(fasttext_model_path):
        # English if_exists=('strict' 'overwrite')
        fasttext.util.download_model('en', if_exists='strict')
        shutil.move(fasttext_model_name, fasttext_model_path)
        print('Model loaded')
    print('Loading fasttext model from %s...' % fasttext_model_path)
    model = fasttext.load_model(fasttext_model_path)
    # print('Model size reducing...')
    # fasttext.util.reduce_model(model, 100)

    return model


def strong_lemmatizing(row, unique=False):
    """
    ['delivery', 'courier', 'delivered', 'shipping', 'shipment', 'delivering', 'deliver', 'dispatch', 'service', 'ordering']
    """
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    pre_tokens = row.split(' ')
    lemmatized_verbs = [lemmatizer.lemmatize(
        word, pos='v') for word in pre_tokens]
    transformed_words = [lemmatizer.lemmatize(
        word, 'n') for word in lemmatized_verbs]
    if unique:
        transformed_words = list(set(lemmatized_verbs))
    return transformed_words


def text_to_pos_tags(text: str):
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    tokens = word_tokenize(text)
    pos_tags = [tag for _, tag in pos_tag(tokens)]
    return ' '.join(pos_tags)


def get_tokens_matix(
        df,
        tokens_matrix_path,
        overwrite=False,
        col_name='content_preprocessed'):
    if overwrite:
        print('current version removed')
        os.remove(tokens_matrix_path)
    tokens_matrix = np.array([])
    if os.path.exists(tokens_matrix_path):
        print('Loading...')
        tokens_matrix = np.load(tokens_matrix_path, allow_pickle=True)
    else:
        print('Transforming...')
        tokens_matrix = df[col_name].apply(strong_lemmatizing).values
        print('Saving...')
        np.save(tokens_matrix_path, tokens_matrix)
    return tokens_matrix


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


def get_ngrams(tokens_matrix, window: int):
    ngrams_flatten = [
        ' '.join(
            sorted(ngram)) for sublist in tokens_matrix for ngram in ngrams(
            sublist,
            window)]
    res = Counter(ngrams_flatten)
    return res


def generate_ngram_tags(tokens, max_ngram_range: int = 3):
    from nltk.util import ngrams

    res = []
    for window in range(2, max_ngram_range + 1):
        ngrams_flatten = [' '.join(sorted(ngram))
                          for ngram in ngrams(tokens, window)]
        res += ngrams_flatten
    return res


def generate_ngram_tags(tokens, max_ngram_range: int = 3):
    from nltk.util import ngrams
    res = []
    for window in range(2, max_ngram_range + 1):
        ngrams_flatten = [' '.join(sorted(ngram))
                          for ngram in ngrams(tokens, window)]
        res += ngrams_flatten
    return res


def find_closest_tag_levenstein(tag_name, candidates_array):
    from Levenshtein import distance

    item_distances = [int(100 *
                          distance(tag_name.lower(), j.lower()) /
                          len(tag_name)) for j in candidates_array]
    catalog_entities_sim = np.argsort(item_distances)[:10]
    print(tag_name, [(candidates_array[k], item_distances[k])
                     for k in catalog_entities_sim])
    return catalog_entities_sim


def top_similar_tags(tag_id, candidates_array, candidates_embeds, top=10):
    from sklearn.metrics.pairwise import cosine_similarity

    query_embed = candidates_embeds[tag_id, :]
    tag_name = candidates_array[tag_id]
    sims = cosine_similarity(query_embed.reshape(1, -1), candidates_embeds)[0]
    print(len(sims))
    top_similar_idx = [int(i) for i in np.argsort(-np.abs(sims))][:top]
    print(tag_name)
    res = dict([(candidates_array[k], sims[k]) for k in top_similar_idx])
    return res

#-------- FAST TEXT EMBEDDER -------- #
def get_tokens(line):
    tokens = line.split(' ')
    return tokens

def get_word_2_vec_vector(tokens, model):
    return np.asarray([model[el] for el in tokens]).sum(axis=0)

def get_normed_vector(raw_txt, model):
    vec = get_word_2_vec_vector(get_tokens(raw_txt), model)
    vec_norm = np.linalg.norm(vec)
    vec /= vec_norm

    return vec.astype(np.float32)

def get_fasttext_corpus_vectors(df, model, synopsis_col: str = "synopsis"):
    import tqdm

    corpus_vectors = np.zeros(shape=(df.shape[0], 300)).astype(np.float32)
    for raw, data in tqdm.tqdm(df.iterrows()):
        vec = get_normed_vector(data[synopsis_col], model)
        corpus_vectors[raw, :] = vec
    corpus_vectors = corpus_vectors.astype(np.float32)
    return corpus_vectors

def eval_embeds(df, txt_col, npy_filename: str, root_data_dir: str, overwrite: bool = True):
  if overwrite:
    if os.path.exists(npy_filename):
      os.remove(npy_filename)
  if os.path.exists(npy_filename):
    print('Loading from a dump')
    corpus_numpy = np.load(npy_filename)
  else:
    # Warning: it takes 20 minutes to download the model!
    print('Model loading...')
    model = prepare_fasttext(root_data_dir)
    print('Eval embeddings...')
    corpus_numpy = get_fasttext_corpus_vectors(df, model, synopsis_col=txt_col)
    print('Saving file %d rows to %s...' % (corpus_numpy.shape[0], npy_filename))
    np.save(npy_filename, corpus_numpy)
  return corpus_numpy

def get_fasttext_vectors(candidates_list, model):
    import tqdm

    corpus_vectors = np.zeros(shape=(len(candidates_list), 300)).astype(np.float32)
    for raw, data in tqdm.tqdm(enumerate(candidates_list)):
        vec = get_normed_vector(data, model)
        corpus_vectors[raw, :] = vec
    corpus_vectors = corpus_vectors.astype(np.float32)
    return corpus_vectors

def concat_batches(batches_dir):
  res = []
  for f in os.listdir(batches_dir):
    if '.npy' in f:
      res.append(np.load(os.path.join(batches_dir, f)))
  res = np.concatenate(res, axis=0)
  return res
