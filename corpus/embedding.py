
import numpy as np
from .common import glove,embedding_dim
from gensim.models import Word2Vec


_MODEL = Word2Vec.load_word2vec_format(glove(), binary=False)
_MODEL.init_sims(replace=True)

def get_model():
    return _MODEL

class MeanEmbeddingVectorizer(object):
    word2vec = _MODEL
    dim = embedding_dim()

    def fit(self, X, y):
        return self

    def transform(self, X, Weights = False):
        if not Weights:
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        else:
            return np.array([
                np.mean([self.word2vec[w] * Weights[i][j] for j,w in enumerate(words) if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for i,words in enumerate(X)
            ])