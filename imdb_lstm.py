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


from tqdm import tqdm

np.random.seed(1337)  # for reproducibility


from corpus.embedding import Generator
from sklearn import metrics
from corpus.networks import rnn, cnn, cnn_rnn

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

print('Loading data...')

Generator_BATCH=32

gen_train = Generator(multiprocessing=10,categorical=True,local=False)
gen_train.yeld_size=Generator_BATCH
gen_train.calculate_embedding()


gen_test = Generator(multiprocessing=10,yeld_corpus="test",categorical=True,local=False)
gen_test.yeld_size=Generator_BATCH

model_choice='cnn-rnn'



print('Build model...')
models= {'cnn':cnn,'cnn-rnn':cnn_rnn,'rnn':rnn}
model=models[model_choice](256, len(gen_train.imdb_vocab)-1,gen_train.maxlen,embedding_matrix=gen_train.embedding_matrix,embedding_trainable=True,lstm_dropout=.0)



# model = Sequential()
# model.add(LSTM(256,input_dim=88584, dropout_W=.0, dropout_U=.0, return_sequences=False))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


model.summary()

print('Train...')

# model.fit_generator(gen_train(),samples_per_epoch=25000,nb_epoch=2)


model.fit(gen_train.x_train, gen_train.train_y, batch_size=128, nb_epoch=2,
          validation_data=(gen_test.x_test, gen_test.test_y))





# y_output=np.zeros((25000,1))
# for batch,(x,y) in tqdm(enumerate(gen_test()),total=int(25000/Generator_BATCH)):
#     y_output[(batch*Generator_BATCH):(batch*Generator_BATCH)+Generator_BATCH] = model.predict(x)




y_output = model.predict(gen_test.x_test, 128)
y_output[np.where(y_output<0.5)]=0
y_output[np.where(y_output>=0.5)]=1

accuracy = metrics.accuracy_score(gen_test.test_y, y_output)
precision = metrics.precision_score(gen_test.test_y, y_output)
recall = metrics.recall_score(gen_test.test_y, y_output)
f1 = metrics.f1_score(gen_test.test_y, y_output)

print("Accuracy: %f\nPrecission: %f\nRecall: %f\nF1: %f" % (accuracy,precision,recall,f1))
