from keras.datasets import imdb
from keras.preprocessing import sequence

from .common import glove, embedding_dim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from os.path import join
from os import listdir
import numpy as np
from tqdm import *
import os
from multiprocessing import Pool
from keras.utils import np_utils
from keras.backend.common import floatx

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

class Generator():

    def __init__(self,vocab=None,maxlen=80,train_y=None,test_y=None,train_dir=None,
                 test_dir=None,local=False,yeld_corpus="train",multiprocessing=8, categorical=False):
        self._categorical=categorical
        self._maxlen=maxlen
        self._multiprocessing=multiprocessing
        self._yeld_corpus = yeld_corpus
        if not local:
            self._imdb_vocab = imdb.get_word_index()
            if yeld_corpus=="train":
                (self._x_train, self._train_y), (_, _) = imdb.load_data(nb_words=len(self._imdb_vocab.keys()) + 1)
                self._x_train = sequence.pad_sequences(self._x_train, maxlen=maxlen)
                self._x_train[:,1:]=self._x_train[:,1:]-3
            else:
                (_, _), (self._x_test, self._test_y) = imdb.load_data(nb_words=len(self._imdb_vocab.keys()) + 1)
                self._x_test = sequence.pad_sequences(self._x_test, maxlen=maxlen)
                self._x_test[:,1:]=self._x_test[:,1:]-3
        else:
            self._imdb_vocab = open(vocab, encoding="utf8").read().split("\n")
            if not os.path.exists("x_data.npz"):
                self._train_y=train_y
                self._test_y=test_y
                self._train_dir=train_dir
                self._test_dir=test_dir

                self._file_count_train = sum(map(lambda dir_: len(listdir(dir_)), train_dir))
                self._file_count_test = sum(map(lambda dir_: len(listdir(dir_)), test_dir))
                self._vectorize()
            else:
                data=np.load("x_data.npz")
                index_array_train = np.random.permutation(data['x_train'].shape[0])
                index_array_test = np.random.permutation(data['x_test'].shape[0])
                if yeld_corpus=="train":
                    self._train_y=data['y_train'][index_array_train]
                    self._x_train=data['x_train'][index_array_train]
                else:
                    self._test_y=data['y_test'][index_array_test]
                    self._x_test=data['x_test'][index_array_test]
                del(data)
    def calculate_embedding(self):
        inverted_imdb_vocab = dict((v,k) for k,v in self.imdb_vocab.items())
        self.embedding_matrix=np.array([get_model()[inverted_imdb_vocab[w]] if inverted_imdb_vocab[w]
                                                    in get_model().vocab else np.zeros((embedding_dim()),dtype=floatx())
                                                    for w in range(1,len(inverted_imdb_vocab))],dtype=floatx())

    @property
    def maxlen(self):
        return self._maxlen
    @maxlen.setter
    def maxlen(self, maxlen):
        self._maxlen = maxlen

    @property
    def yeld_size(self):
        return self._yeld_size
    @yeld_size.setter
    def yeld_size(self, yeld):
        self._yeld_size = yeld
    
    @property
    def imdb_vocab(self):
        return self._imdb_vocab
    @imdb_vocab.setter
    def imdb_vocab(self,vocab):
        self._imdb_vocab=vocab
    
    @property
    def x_train(self):
        return self._x_train
    @x_train.setter
    def x_train(self,x_train):
        self._x_train=x_train   
        
        
    @property
    def train_y(self):
        return self._train_y
    @train_y.setter
    def train_y(self,train_y):
        self._train_y=train_y
    

    @property
    def x_test(self):
        return self._x_test
    @x_test.setter
    def x_test(self,x_test):
        self._x_test=x_test   
        
        
    @property
    def test_y(self):
        return self._test_y
    @test_y.setter
    def test_y(self,test_y):
        self._test_y=test_y

    

    def multiprocess_file(self,path):
        with open(path,'r', encoding="utf8") as f:
            words = []
            for word in word_tokenize(f.read().lower()):
                try:
                    index = self._imdb_vocab.index(word)
                    words.append(index)
                except:pass
            return words

    def walk(self,train_dir):
        for dir_ in train_dir:

            with Pool(self._multiprocessing) as pool:
                files=[]
                for doc,filename in enumerate(listdir(dir_)):
                    if(len(files)<self._multiprocessing):files.append(join(dir_, filename))
                    else:
                        res = pool.map(self.multiprocess_file,files)
                        files=[]
                        for result in res:
                            yield np.array(result,dtype='int')
                if not len(files)==0:
                    res = pool.map(self.multiprocess_file,files)
                    files=[]
                    for result in res:
                        yield np.array(result,dtype='int')




    def _vectorize(self):

        x_train=np.zeros((self._file_count_train, self._maxlen), dtype='int')
        x_test=np.zeros((self._file_count_test, self._maxlen), dtype='int')

        for doc,document in tqdm(enumerate(self.walk(train_dir)),desc='Train'):
            if document.shape[0]>=maxlen:
                x_train[doc]=document[:maxlen]
            else:
                x_train[doc,:document.shape[0]]=document
        self._x_train=x_train

        for doc,document in tqdm(enumerate(self.walk(test_dir)),desc='Test'):
            if document.shape[0]>=maxlen:
                x_test[doc]=document[:maxlen]
            else:
                x_test[doc,:document.shape[0]]=document
        self._x_test=x_test
        np.savez("x_data.npz", x_train=x_train, x_test=x_test, y_train=self._train_y, y_test=self._test_y)
        if not self._yeld_corpus== "train":
            del(self._train_y)
            del(self._x_train)
        else:
            del(self._test_y)
            del(self._x_test)

    def __call__(self):
        if self._categorical:
            max_features=len(self._imdb_vocab)
            if self._yeld_corpus=="train":
                X_train_=np.zeros((self._yeld_size,self._maxlen,max_features))
                for j in range(int(25000/self._yeld_size)):
                    for i in range(self._yeld_size):
                        X_train_[(self._yeld_size*j)+i,:,:] = np_utils.to_categorical(self._x_train[(self._yeld_size * j) + i, :], nb_classes=max_features)
                        yield X_train_, self._train_y[self._yeld_size * j:self._yeld_size * (j + 1)]
            else:
                X_test_=np.zeros((self._yeld_size,self._maxlen,max_features))
                for j in range(int(25000/self._yeld_size)):
                    for i in range(self._yeld_size):
                        X_test_[(self._yeld_size*j)+i,:,:] = np_utils.to_categorical(self._x_test[(self._yeld_size * j) + i, :], nb_classes=max_features)
                        yield X_test_, self._test_y[self._yeld_size * j:self._yeld_size * (j + 1)]
        else:
            if self._yeld_corpus=="train":
                for j in range(int(25000/self._yeld_size)):
                    yield self._x_train[self._yeld_size * j:self._yeld_size * (j + 1), :], self._train_y[self._yeld_size * j:self._yeld_size * (j + 1)]
            else:
                for j in range(int(25000/self._yeld_size)):
                    yield self._x_test[self._yeld_size * j:self._yeld_size * (j + 1), :], self._test_y[self._yeld_size * j:self._yeld_size * (j + 1)]





if __name__=='__main__':
    train_dir = ["aclimdb/train/neg","aclimdb/train/pos"]
    test_dir = ["aclimdb/test/neg","aclimdb/test/pos"]
    train_y = np.zeros((25000))
    test_y = np.zeros((25000))
    train_y[len(listdir("aclimdb/train/neg")):]=1
    test_y[len(listdir("aclimdb/test/neg")):]=1
    yeld_size=32
    maxlen=80

    vocab = "aclimdb/imdb.vocab"
    gen = Generator(vocab,maxlen,train_y,test_y,train_dir,test_dir,multiprocessing=10)
    gen.yeld_size=100
    for x,y in gen():
        print(x)
        print(y)
        print(x.shape,y.shape)