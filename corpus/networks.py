from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from corpus.common import embedding_dim

def rnn(width,embedding_size,maxlen,embedding_dropout=0.2, embedding_matrix=None,embedding_trainable=True,lstm_dropout=0.2):
    embedding_dims=embedding_dim()

    model = Sequential()
    if embedding_matrix is None:
        model.add(Embedding(embedding_size, embedding_dims,
                            dropout=embedding_dropout,trainable=embedding_trainable))
    else:
        model.add(Embedding(embedding_size, embedding_dims,
                            dropout=embedding_dropout,trainable=embedding_trainable,weights=[embedding_matrix]))

    model.add(LSTM(width, dropout_W=lstm_dropout, dropout_U=lstm_dropout, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn(hidden_dims,embedding_size,maxlen,nb_filter = 250,filter_length = 3,
        embedding_dropout=0.2, embedding_matrix=None,embedding_trainable=True):
    model = Sequential()
    embedding_dims=embedding_dim()


    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    if embedding_matrix is None:
        model.add(Embedding(embedding_size,
                        embedding_dims,
                        input_length=maxlen,
                        dropout=embedding_dropout,trainable=embedding_trainable))
    else:
        model.add(Embedding(embedding_size,
                        embedding_dims,
                        input_length=maxlen,
                        dropout=embedding_dropout,weights=[embedding_matrix],trainable=embedding_trainable))
    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_rnn(width,embedding_size,maxlen,filter_length = 5,nb_filter = 64,
            pool_length = 4,embedding_dropout=0., embedding_matrix=None,embedding_trainable=True,lstm_dropout=0.):
    embedding_dims=embedding_dim()

    model = Sequential()
    if embedding_matrix is None:
        model.add(Embedding(embedding_size, embedding_dims, input_length=maxlen,
                            dropout=embedding_dropout,trainable=embedding_trainable))
    else:
        model.add(Embedding(embedding_size, embedding_dims, input_length=maxlen,
                            dropout=embedding_dropout,weights=[embedding_matrix],trainable=embedding_trainable))
    # model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(width,dropout_W=lstm_dropout, dropout_U=lstm_dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model