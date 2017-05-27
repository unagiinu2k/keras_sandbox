#https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, , Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

import numpy
if False:
    print(X_train[0])#sequence of numbers (= corresponding to words in the vocab)
    max([len(x) for x in X_train])#2494
    max([len(x) for x in X_test])#2315
min([len(x) for x in X_train])#11
    min([len(x) for x in X_test])#7
    numpy.mean([len(x) for x in X_train])#238.71364
    numpy.mean([len(x) for x in X_test])#230.80420
    train_len = [len(x) for x in X_train]
    numpy.argmin(train_len)#6719



print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
if False:
    X_train[6719]#zeros are inserted at the beginning of the sequence
    from itertools import chain
    len(set(chain.from_iterable(X_train)) | set(chain.from_iterable(X_test))) #19932
    len(set(chain.from_iterable(numpy.concatenate((X_train , X_test) , axis = 0)))) #19932x

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))