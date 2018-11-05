import sys
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import cross_val_score
# import nltk

MAX_LEN = 50
MIN_LEN = 3
UNIT = "word"

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

def train_svm(X_train, y_train):
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(twenty_train.data)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # TODO: do n-grams and add NLTK features! 

    # using default SVM params
    # penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, 
    # multi_class=’ovr’, fit_intercept=True, intercept_scaling=1, 
    # class_weight=None, verbose=0, random_state=None, max_iter=1000

    text_clf_svm = Pipeline([('vect', CountVectorizer()), 
                         ('tfidf', TfidfTransformer()),
                         ('svm', LinearSVC())])

    text_clf_svm = text_clf_svm.fit(X_train, y_train)
    # TODO: save model
    return text_clf_svm

def predict(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred

def print_metrics(y_test, y_pred):
    print(precision_recall_fscore_support(y_test, y_pred))
    print("accuracy = ", accuracy_score(y_test, y_pred))

def run_svm(X_train, y_train, X_test, y_test):
    model = train_svm(X_train, y_train)
    y_pred = predict(X_test, model)
    print_metrics(y_test, y_pred)
    scores = cross_val_score(model, X_train + X_test, y_train + y_test, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s train_file_name test_file_name save_model" % sys.argv[0])
    X_train, y_train, X_test, y_test = load_data()
    run_svm(X_train, y_train, X_test, y_test)
    


