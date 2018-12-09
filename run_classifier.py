import sys
import re
import pickle as pkl
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import LatentDirichletAllocation

# import nltk
import nltk
from nltk.corpus import stopwords as sw
import nltk.stem.snowball as sb
import numpy as np
import scipy.sparse as sparse

MAX_LEN = 50
MIN_LEN = 3
UNIT = "word"

class StemTokenizer:
    def __init__(self):
        self.stemmer = sb.SnowballStemmer('english')
    
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in nltk.word_tokenize(doc)]

#input for this class will be a tuple: words and stylometric features
class AugmentedCountVectorizer:
    
    def __init__(self, input='content', encoding='utf-8', decode_error='strict',
                 strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
                 binary=False, dtype=np.int64, useStyleFeatures=False):
        
        self._cv = CountVectorizer(input, encoding, decode_error, strip_accents, lowercase,
                             preprocessor, tokenizer, stop_words, token_pattern, ngram_range,
                             analyzer, max_df, min_df, max_features, vocabulary, binary, dtype)
        self._useStyleFeatures = useStyleFeatures
    
    def fit(self, X, y=None):
        self._cv.fit(X[0])
        return self
    
    def transform(self, X):
        Xnew = self._cv.transform(X[0])
        if self._useStyleFeatures:
            Xret = (Xnew, np.nan_to_num(np.vstack(X[1])))
        else:
            Xret = Xnew
        return Xret

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class RemoveWords:
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.nan_to_num(np.vstack(X[1]))
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


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

class AllFeatureTransformer():
    
    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, learning_method='batch', learning_decay=0.7,
                 learning_offset=10.0, max_iter=10, batch_size=128,
                 evaluate_every=-1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None, n_topics=None):
        self._tfidf = TfidfTransformer(norm, use_idf, smooth_idf, sublinear_tf)
        self._topicModel = LatentDirichletAllocation(n_components, doc_topic_prior, topic_word_prior, learning_method, learning_decay, learning_offset, max_iter, batch_size, evaluate_every, total_samples, perp_tol, mean_change_tol, max_doc_update_iter, n_jobs, verbose, random_state, n_topics)
    
    def fit(self, X, y=None):
        self._tfidf.fit(X[0])
        self._topicModel.fit(X[0])
        self._styleData = X[1]
        return self
    
    def transform(self, X, copy=True):
        temp1 = self._tfidf.transform(X[0], copy)
        temp2 = self._topicModel.transform(X[0])
        temp3 = np.nan_to_num(np.vstack(X[1]))
        print(temp1.shape)
        print(type(temp1))
        print(temp2.shape)
        print(type(temp2))
        return sparse.hstack([temp1, sparse.csr_matrix(temp2), sparse.csr_matrix(temp3)])
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# get data
def load_data():
    X_train, s_train, y_train, X_test, s_test, y_test = [], [], [], [], [], []
    # load train
    fo = open(sys.argv[2])
    for line in fo:
        if len(line.split("\t")) < 2:
            continue
        s = np.zeros((14))
        x, y, *s = line.split("\t")
        x = tokenize(x, UNIT)
        s[-1] = s[-1].strip()
        for ii in [0, 1, 3]:
            s[ii] = int(s[ii])
        for ii in [2]+list(range(4,14)):
            s[ii] = float(s[ii])
        if len(x) < MIN_LEN:
            continue
        # x = x[:MAX_LEN]
        x = " ".join(x)
        X_train.append(x)
        s_train.append(np.array(s))
        y_train.append(y)

    # load test
    fo = open(sys.argv[3])
    for line in fo:
        if len(line.split("\t")) < 2:
            continue
        s = np.zeros((14))
        x, y, *s = line.split("\t")
        x = tokenize(x, UNIT)
        s[-1] = s[-1].strip()
        for ii in [0, 1, 3]:
            s[ii] = int(s[ii])
        for ii in [2]+list(range(4,14)):
            s[ii] = float(s[ii])
        if len(x) < MIN_LEN:
            continue
        # x = x[:MAX_LEN]
        x = " ".join(x)
        X_test.append(x)
        s_test.append(np.array(s))
        y_test.append(y)

    return X_train, s_train, y_train, X_test, s_test, y_test

def train(X_train, s_train, y_train):
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(twenty_train.data)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # TODO: do n-grams and add NLTK features! 

    # using default SVM params
    # penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, 
    # multi_class=’ovr’, fit_intercept=True, intercept_scaling=1, 
    # class_weight=None, verbose=0, random_state=None, max_iter=1000
    

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
    if sys.argv[4].lower()=='true':
        tkzr = StemTokenizer()
    else:
        tkzr = None
    
    if sys.argv[5].lower()!='true':
        swlist = []
    
    if sys.argv[1].lower()=='rf':
        classTuple = ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
    elif sys.argv[1].lower()=='svm':
        classTuple = ('svm', LinearSVC(class_weight='balanced'))
    elif sys.argv[1].lower()=='knn':
        classTuple = ('knn', KNeighborsClassifier(n_neighbors=5, metric='cosine'))
    else:
        sys.exit('unknown classifier')

    #what features are we using?
    if sys.argv[7].lower()=='word':
        text_clf = Pipeline([('vect', AugmentedCountVectorizer(stop_words=swlist,
                                                               tokenizer=tkzr)),
                             ('tfidf', TfidfTransformer()),
                             classTuple])
    elif sys.argv[7].lower()=='topic':
        text_clf = Pipeline([('vect', AugmentedCountVectorizer(stop_words=swlist,
                                                               tokenizer=tkzr)),
                             ('tfidf', LatentDirichletAllocation(n_components=50)),
                             classTuple])
    elif sys.argv[7].lower()=='style':
        text_clf = Pipeline([('vect', RemoveWords()), classTuple])
    elif sys.argv[7].lower()=='all':
        text_clf = Pipeline([('vect', AugmentedCountVectorizer(stop_words=swlist,
                                                               tokenizer=tkzr,
                                                               useStyleFeatures=True)),
                             ('tfidf', AllFeatureTransformer(n_components=50)),
                             classTuple])
    else:
        sys.exit('unknown features')
    

    text_clf = text_clf.fit((X_train, s_train), y_train)
    # TODO: save model
    return text_clf

def predict(X_test, s_test, model):
    y_pred = model.predict((X_test, s_test))
    return y_pred

def print_metrics(y_test, y_pred):
    p, r, fs, s = precision_recall_fscore_support(y_test, y_pred)
    print((p, r, fs, s))
    print("accuracy = ", accuracy_score(y_test, y_pred))
    f = open(sys.argv[6], mode='wb')
    pkl.dump((p, r, fs, s), f)
    f.close()

def run_classifier(X_train, s_train, y_train, X_test, s_test, y_test):
    model = train(X_train, s_train, y_train)
    y_pred = predict(X_test, s_test, model)
    print_metrics(y_test, y_pred)
    #scores = cross_val_score(model, X_train + X_test, y_train + y_test, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == "__main__":
    if len(sys.argv) != 8:
        sys.exit("Usage: %s classifier_name train_file_name test_file_name use_stemming remove_stopwords save_model features" % sys.argv[0])
    X_train, s_train, y_train, X_test, s_test, y_test = load_data()
    run_classifier(X_train, s_train, y_train, X_test, s_test, y_test)
    


