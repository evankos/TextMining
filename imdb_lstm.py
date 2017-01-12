'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np


np.random.seed(1337)  # for reproducibility


from corpus.embedding import Generator
from sklearn import metrics
from corpus.networks import rnn, cnn, cnn_rnn



print('Loading data...')


gen_train = Generator(multiprocessing=10,categorical=True,local=False)
gen_train.yeld_size=100
gen_train.calculate_embedding()


gen_test = Generator(multiprocessing=10,yeld_corpus="test",categorical=True,local=False)
gen_test.yeld_size=100

model_choice='rnn'



print('Build model...')
models= {'cnn':cnn,'cnn-rnn':cnn_rnn,'rnn':rnn}
model=models[model_choice](256, len(gen_train.imdb_vocab)-1,gen_train.maxlen,embedding_matrix=gen_train.embedding_matrix,embedding_trainable=False,lstm_dropout=0.1)

model.summary()
exit()
print('Train...')
# model.fit_generator(gen_train,samples_per_epoch=25000,nb_epoch=10)


model.fit(gen_train.x_train, gen_train.train_y, batch_size=128, nb_epoch=25,
          validation_data=(gen_test.x_test, gen_test.test_y))

y_output = model.predict(gen_test.x_test, 128)
y_output[np.where(y_output<0.5)]=0
y_output[np.where(y_output>=0.5)]=1

accuracy = metrics.accuracy_score(gen_test.test_y, y_output)
precision = metrics.precision_score(gen_test.test_y, y_output)
recall = metrics.recall_score(gen_test.test_y, y_output)
f1 = metrics.f1_score(gen_test.test_y, y_output)

print("Accuracy: %f\nPrecission: %f\nRecall: %f\nF1: %f" % (accuracy,precision,recall,f1))
