from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from corpus.dataset import get_data_feat,tf_idf
from corpus.embedding import MeanEmbeddingVectorizer
import numpy as np
from tqdm import *
from nltk.corpus import stopwords
from numpy.linalg import norm


batch_size=5000
apply_tf_idf=False
word2vec=True
remove_stopwords=True

stops = set(stopwords.words('english'))
imdb_vocab = open("aclimdb/imdb.vocab", encoding="utf8").read().split("\n")



#Get stop word indexes
stop_indices=[i for i,word in enumerate(imdb_vocab) if word in stops]

X_train, y_train = get_data_feat("aclimdb/train/labeledBow.feat")
X_test, y_test = get_data_feat("aclimdb/train/labeledBow.feat")


# Setting positive and negative classes only
y_train[np.where(y_train<7)]=0
y_train[np.where(y_train>=7)]=1

y_test[np.where(y_test<7)]=0
y_test[np.where(y_test>=7)]=1

c=np.unique(y_train)



if word2vec:
    import corpus.embedding as E
    embedding = E.get_model()
    classifier = GaussianNB()
    vectorizer = MeanEmbeddingVectorizer()
else:
    classifier = MultinomialNB()

for batch in tqdm(range(int(25000/batch_size))):
    x = (X_train.toarray())[batch*batch_size:batch*batch_size+batch_size]

    # Remove stopwords
    if remove_stopwords:
        x[:,stop_indices] = 0

    # Apply TF-IDF on batch
    if apply_tf_idf:
        x=tf_idf(x)

    if word2vec:
        words = [[imdb_vocab[i] for i in np.where(x[doc]>0)[0]] for doc in range(x.shape[0])]
        weights = [x[doc,np.where(x[doc]>0)[0]] for doc in range(x.shape[0])]
        if apply_tf_idf:xx = vectorizer.transform(words, weights)
        else:xx = vectorizer.transform(words)
        x=xx

    classifier.partial_fit(x, y_train[batch*batch_size:batch*batch_size+batch_size],classes=c)




for batch in tqdm(range(int(25000/batch_size))):
    x = (X_test.toarray())[batch*batch_size:batch*batch_size+batch_size]

    # Remove stopwords
    if remove_stopwords:
        x[:,stop_indices] = 0

    # Apply TF-IDF on batch
    if apply_tf_idf:
        x=tf_idf(x)

    if word2vec:
        words = [[imdb_vocab[i] for i in np.where(x[doc]>0)[0]] for doc in range(x.shape[0])]
        weights = [x[doc,np.where(x[doc]>0)[0]] for doc in range(x.shape[0])]
        if apply_tf_idf:xx = vectorizer.transform(words, weights)
        else:xx = vectorizer.transform(words)
        x=xx

    if batch==0:y_output = classifier.predict(x)
    else:y_output = np.concatenate((y_output,classifier.predict(x)),axis=0)


accuracy = metrics.accuracy_score(y_test, y_output)
precision = metrics.precision_score(y_test, y_output)
recall = metrics.recall_score(y_test, y_output)
f1 = metrics.f1_score(y_test, y_output)

print("Accuracy: %f\nPrecission: %f\nRecall: %f\nF1: %f" % (accuracy,precision,recall,f1))



