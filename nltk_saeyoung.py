import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from collections import Counter
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pprint import pprint

from sklearn.feature_extraction import DictVectorizer

def get_continuous_chunks(text):
    #input: document text
    #output: list of the actual words (named entities)
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
            else:
                    continue
    return continuous_chunk

def num_named_entity(text):
    #input: document text
    #output: int
    named_entity = np.array(get_continuous_chunks(text))
    return named_entity.shape[0]

def num_sen(X):
    #input: document text
    #output: int
    sentences = len(X.split("."))
    return sentences

def num_word(X):
    #input: document text
    #output: int
    words = len(X.split())
    return words

def avg_word_per_sen(X):
    #input: document text
    #output: float
    word_per_sen = num_word(X)/num_sen(X)
    return word_per_sen

def frac_words(X):
    #input: document text
    #output: (1,2) np.array
    
    num_of_sen = np.array(X.split(".")).shape[0]
    
    #tokenize using NLTK
    tokens = nltk.word_tokenize(X)
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    
    #count words and nouns and normalize by total counts
    counts = Counter(tag for word,tag in tags)
    total = sum(counts.values())
    frac = dict((word, float(count)/total) for word,count in counts.items())
    
    #frac_noun: fraction of nouns
    noun_tags = ["NN","NNS","NNP","NNPS","PRP","PRP$","WP","WP$","WRB"]
    frac_noun = 0
    for tag_i in noun_tags:
        if tag_i in frac:
            frac_noun += frac[str(tag_i)]
            
    #frac_verb: fraction of verbs
    verb_tags = ["RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]
    frac_verb = 0
    for tag_i in verb_tags:
        if tag_i in frac:
            frac_verb += frac[str(tag_i)]
    
    return np.array([[frac_noun, frac_verb]])

#### NLTK Tags including nouns
# NN    noun, singular 'desk'
# NNS   noun plural 'desks'
# NNP   proper noun, singular   'Harrison'
# NNPS  proper noun, plural 'Americans'
# PRP   personal pronoun    I, he, she
# PRP$  possessive pronoun  my, his, hers
# WP    wh-pronoun  who, what
# WP$   possessive wh-pronoun   whose
# WRB   wh-abverb   where, when

#### NLTK Tags including verbs
# RB    adverb  very, silently,
# RBR   adverb, comparative better
# RBS   adverb, superlative best
# VB    verb, base form take
# VBD   verb, past tense    took
# VBG   verb, gerund/present participle taking
# VBN   verb, past participle   taken
# VBP   verb, sing. present, non-3d take
# VBZ   verb, 3rd person sing. present  takes

analyser = SentimentIntensityAnalyzer()

def polarity_score_by_sentence(text):
    #input: document text
    #output: pandas df, #rows:number of sentences, #columns: 'neg','neu','pos','compound'
    
    sentence_list = text.split(".")
    pscore_list = pd.DataFrame(index = range(1,num_sen(text)+1),columns=['neg','neu', 'pos', 'compound'])

    for i in range(num_sen(text)):
        pscore = analyser.polarity_scores(sentence_list[i])
        pscore_list.loc[i+1,:] = pscore
    return pscore_list

def polarity_scores(text):
    #input: document text
    #output: (1,8) np.array, first four-avg, last four-variance
    #four numbers corresponding to'neg','neu','pos','compound'
    
    pscore_list = polarity_score_by_sentence(text)
    
    return np.append(pscore_list.mean(),pscore_list.var())



#making dataframe

def make_stylo_features(X):
    #input: document text
    #output: dataframe, one row (for one article)
    Y = np.array([[num_sen(X),num_word(X), avg_word_per_sen(X), num_named_entity(X)]])
    Y = np.concatenate((Y,frac_words(X),polarity_scores(X)), axis=None)
    
    Y = pd.DataFrame(np.array([Y]), columns= ["num_sen","num_word", "avg_word_per_sen", "num_named_entity","frac_words_noun","frac_words_verb",
                    "avg_polarity_score_negative","avg_polarity_score_neutral","avg_polarity_score_positive","avg_polarity_score_compound",
                    "var_polarity_score_negative","var_polarity_score_neutral","var_polarity_score_positive","var_polarity_score_compound"])  
    return Y

def make_stylo_features_df(data):
    #input: train/test dataset (dataframe)
    #output: dataframe of stylo features
    #apply make_stylo_features row by row and return the df of size (num_row_dataset x 14)
    stylo_features = make_stylo_features(data.loc[1,"content"])
    for i in range(1,data.shape[0]):
        X = data.loc[i,"content"]
        stylo_features = pd.concat([stylo_features,make_stylo_features(X)], axis=0, ignore_index=True)
    return stylo_features

def add_stylo_features(data):
    #input: train/test dataset (dataframe)
    #output: dataframe
    #add stylo feature columns to the original dataset
    stylo_features = make_stylo_features(data.loc[1,"content"])
    for i in range(1,data.shape[0]):
        X = data.loc[i,"content"]
        stylo_features = pd.concat([stylo_features,make_stylo_features(X)], axis=0, ignore_index=True)
    data = pd.concat([data,stylo_features], axis=1)
    return data


# load data from csv files
# sample=pd.read_csv('sample_article.csv',header=0)
articles1_df = pd.read_csv("data/articles1.csv")
articles2_df = pd.read_csv("data/articles2.csv")
articles3_df = pd.read_csv("data/articles3.csv")
articles_df = pd.concat([articles1_df, articles2_df, articles3_df],ignore_index=True)

#sample data for the test
# onesample = sample.ix[1,"content"]

print("Num sentences in total: ", num_sen(onesample))
print("Num words in total: ", num_word(onesample))
print("average word per sentence: ", avg_word_per_sen(onesample))
print("Num named entity: ",num_named_entity(onesample))
print("fraction of words: (noun, verb)", frac_words(onesample))
print("average of the polarity scores: ('neg','neu','pos','compound') =",avg_polarity_score(onesample))
print("variance of the polarity scores: ('neg','neu','pos','compound') =",var_polarity_score(onesample))


if __name__ == "__main__":
    stylo_data_1 = make_stylo_features_df(articles1_df)
    stylo_data_1.to_csv('stylo_data_1.csv', encoding='utf-8')
    stylo_data_2 = make_stylo_features_df(articles2_df)
    stylo_data_2.to_csv('stylo_data_2.csv', encoding='utf-8')
    stylo_data_3 = make_stylo_features_df(articles3_df)
    stylo_data_3.to_csv('stylo_data_3.csv', encoding='utf-8')


