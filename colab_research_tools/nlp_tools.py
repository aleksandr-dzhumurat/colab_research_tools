import os
import shutil

from utils import prepare_dirs

def prepare_nltk(root_data_dir='/srv/data'):
    nltk_data_dir = os.path.join(root_data_dir, 'nltk_data')
    prepare_dirs(nltk_data_dir)
    import nltk

    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)  # Lemmas
    nltk.download('omw-1.4', download_dir=nltk_data_dir)  # Lemmas
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)  # POS tags
    nltk.data.path.append(nltk_data_dir) # тут почему-то корневую надо указывать ¯\_(ツ)_/¯

def prepare_fasttext(root_data_dir, fasttext_model_name = 'cc.en.300.bin'):
  import fasttext.util, fasttext

  fasttext_model_path = os.path.join(root_data_dir, fasttext_model_name)
  if not os.path.exists(fasttext_model_path):
    fasttext.util.download_model('en', if_exists='strict')  # English if_exists=('strict' 'overwrite')
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
  lemmatized_verbs = [lemmatizer.lemmatize(word, pos='v') for word in pre_tokens]
  transformed_word = [lemmatizer.lemmatize(word, 'n') for word in lemmatized_verbs]
  if unique:
    lemmatized_verbs = list(set(lemmatized_verbs))
  return lemmatized_verbs

