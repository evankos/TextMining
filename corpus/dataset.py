from gensim.models import Word2Vec
from .common import set_memory,project_root
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.externals.joblib import Memory
from numpy.linalg import norm


mem = Memory("%s/memcache" % project_root())
set_memory(mem)


def embedding(glove_path):
    model = Word2Vec.load_word2vec_format(glove_path, binary=False)
    model.init_sims(replace=True)
    return model

def tf_idf(x):
    for i in range(x.shape[1]):
            x[:,i] = x[:,i] * np.log(x.shape[0]/(1+np.where(x[:,i]>0)[0].shape[0]))
    return x

@mem.cache
def get_data_feat(file):
    data = load_svmlight_file(file)
    return data[0], data[1]