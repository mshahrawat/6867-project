import sys
import re
import numpy as np
import pandas as pd
import random
import math
from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelBinarizer

import nltk
from nltk.corpus import stopwords as sw
import nltk.stem.snowball as sb

import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer

MAX_LEN = 50
MIN_LEN = 3
UNIT = "word"

class StemTokenizer:
    def __init__(self):
        self.stemmer = sb.SnowballStemmer('english')
    
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in nltk.word_tokenize(doc)]

def normalize(x):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        x = re.sub(" ", "", x)
        return list(x)
    if unit == "word":
        return x.split(" ")

# get data
def load_data():
    X_train, y_train, X_test, y_test = [], [], [], []
    # load train
    fo = open(sys.argv[1])
    for line in fo:
        if len(line.split("\t")) < 2:
            continue
        x, y = line.split("\t")
        x = tokenize(x, UNIT)
        y = y.strip()
        if len(x) < MIN_LEN:
            continue
        # x = x[:MAX_LEN]
        x = " ".join(x)
        X_train.append(x)
        y_train.append(y)

    # load test
    fo = open(sys.argv[2])
    for line in fo:
        if len(line.split("\t")) < 2:
            continue
        x, y = line.split("\t")
        x = tokenize(x, UNIT)
        y = y.strip()
        if len(x) < MIN_LEN:
            continue
        # x = x[:MAX_LEN]
        x = " ".join(x)
        X_test.append(x)
        y_test.append(y)

    return X_train, y_train, X_test, y_test

def run_classifer(X_train, y_train, X_test, y_test):
    # s_train = np.array(s_train) # samples x features
    # s_test = np.array(s_test)

    num_labels = 15
    batch_size = 100

    stemmer = sb.SnowballStemmer('english')
    
    swlist = sw.words('english')
    swlist += [stemmer.stem(w) for w in swlist]
    swlist += ["'d", "'s", 'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'might',
               'must', "n't", 'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'veri', 'whi',
               'wo', 'would', 'yourselv'] #complained about not having these as stop words
    pubs = ['buzzfe', 'buzzf', 'npr', 'cnn', 'vox', 'reuter', 'breitbart', 'fox', 'guardian','review', 'theatlant']
    punct = []#[':', '..', '“', '@', '%', ';', '→', ')', '#', '(', '*', '&', '[', ']', '…', '?','—', '‘', '$'] #gonna leave these in for now
    
    swlist += pubs
    swlist += punct
    if sys.argv[3].lower()=='true':
        tkzr = StemTokenizer()
    else:
        tkzr = None
    
    if sys.argv[4].lower()!='true':
        swlist = []


    #what features are we using?
    if sys.argv[6].lower()=='word':
        count_vect = CountVectorizer(stop_words=swlist, tokenizer=tkzr)
        count_vect.fit(X_train)
        X_train = count_vect.transform(X_train)
        X_test = count_vect.transform(X_test)
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(X_train)
        X_train = tfidf_transformer.transform(X_train)
        X_test = tfidf_transformer.transform(X_test)

    elif sys.argv[6].lower()=='topic':
        count_vect = CountVectorizer(stop_words=swlist, tokenizer=tkzr)
        count_vect.fit(X_train)
        X_train = count_vect.transform(X_train)
        X_test = count_vect.transform(X_test)
        lda_model = LatentDirichletAllocation(n_components=10)
        lda_model.fit(X_train)
        X_train = lda_model.transform(X_train)
        X_test = lda_model.transform(X_test)

    elif sys.argv[6].lower()=='style':
        X_train = csr_matrix(s_train)
        X_test = csr_matrix(s_test)

    elif sys.argv[6].lower()=='all':
        count_vect = CountVectorizer(stop_words=swlist, tokenizer=tkzr)
        count_vect.fit(X_train)
        X_train = count_vect.transform(X_train)
        X_test = count_vect.transform(X_test)

        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(X_train)
        X_train_tf = tfidf_transformer.transform(X_train)
        X_test_tf = tfidf_transformer.transform(X_test)
        print(type(X_train_tf))

        lda_model = LatentDirichletAllocation(n_components=10)
        lda_model.fit(X_train)
        X_train_lda = lda_model.transform(X_train)
        X_test_lda = lda_model.transform(X_test)
        print(type(X_train_lda))

        X_train = csr_matrix(sparse.hstack([X_train_tf, csr_matrix(X_train_lda), csr_matrix(s_train)]))
        X_test = csr_matrix(sparse.hstack([X_test_tf, csr_matrix(X_test_lda), csr_matrix(s_test)]))

        print(type(X_train))

        # sparse.save_npz("X_train" + sys.argv[6] + ".npz", X_train)
        # sparse.save_npz("X_test" + sys.argv[6] + ".npz", X_test)

    else:
        sys.exit('unknown features')

    encoder = LabelBinarizer()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    # sparse.save_npz("y_train" + sys.argv[6] + ".npz", y_train)
    # sparse.save_npz("y_test" + sys.argv[6] + ".npz", y_test)

    # load everything back
    # X_train = sparse.load_npz("X_train.npz")

    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(512, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=5,
                        verbose=1,
                        validation_split=0.1)

    model.model.save(sys.argv[5] + '.h5')

    model = keras.models.load_model(sys.argv[5] + '.h5')
    score = model.evaluate(X_test, y_test,
                           batch_size=batch_size, verbose=1)

    print('Test accuracy:', score[1])

    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    # predicted = np.argmax(pred, axis=1)
    p, r, fs, s = precision_recall_fscore_support(y_test, y_pred)
    print(p, r, fs, s)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s classifier_name train_file_name test_file_name use_stemming remove_stopwords save_model features" % sys.argv[0])
    X_train, y_train, X_test, y_test = load_data()
    run_classifer(X_train, y_train, X_test, y_test)


