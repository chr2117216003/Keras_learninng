
from __future__ import print_function
import numpy as np
import h5py
from keras.models import model_from_json

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import cPickle


def trans(str1):
    a = []
    dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a


def createTrainData(str1):
    sequence_num = []
    label_num = []
    for line in open(str1):
        proteinId, sequence, label = line.split(",")
        proteinId = proteinId.strip(' \t\r\n');
        sequence = sequence.strip(' \t\r\n');
        sequence_num.append(trans(sequence))
        label = label.strip(' \t\r\n');
        label_num.append(int(label))

    return sequence_num,label_num



a,b=createTrainData("positive_and_negative.csv")
t = (a, b)
cPickle.dump(t,open("data.pkl","wb"))

def createTrainTestData(str_path, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.25, seed=113,
              start_char=1, oov_char=2, index_from=3):
    X,labels = cPickle.load(open(str_path, "rb"))

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                                      'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])


    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return (X_train, y_train), (X_test, y_test)



# Embedding
max_features = 23
maxlen = 1000
embedding_size = 128

# Convolution
#filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 128
nb_epoch = 100


print('Loading data...')
(X_train, y_train), (X_test, y_test) = createTrainTestData("data.pkl",nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=10,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))

model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

#json_string = model.to_json()
#open('my_model_rat.json', 'w').write(json_string)
#model.save_weights('my_model_rat_weights.h5')
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print('***********************************************************************')


