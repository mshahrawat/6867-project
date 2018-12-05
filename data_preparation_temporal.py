import pandas as pd
import re
import random
import math
import sys
import datetime as dt

def create_dataframe():
	articles1_df = pd.read_csv(sys.argv[1] + "/articles1.csv")
	articles2_df = pd.read_csv(sys.argv[1] + "/articles2.csv")
	articles3_df = pd.read_csv(sys.argv[1] + "/articles3.csv")
	articles_df = pd.concat([articles1_df, articles2_df, articles3_df])
	# print(articles_df.groupby('publication').size())
	return articles_df

def write_windowed_data(df, train_start, train_stop, test_stop):
    #get training and test periods
    print('splitting up train and test')
    train_df = df[df['date'].astype('datetime64')>=train_start]
    train_df = train_df[train_df['date'].astype('datetime64')<=test_stop]
    test_df = train_df[train_df['date'].astype('datetime64')>train_stop]
    train_df = train_df[train_df['date'].astype('datetime64')<=train_stop]
    
    #get documents and labels for training data
    print('writing training data to file')
    all_pubs = list(train_df['publication'].unique())
    article_text = train_df['content'].tolist()
    article_text = list(map(lambda x: re.sub('\s+',' ', x).strip(), article_text))
    publications = train_df['publication'].tolist()
    write_file(article_text, publications, sys.argv[2])
    
    #same for test
    print('writing test data to file')
    all_pubs = list(test_df['publication'].unique())
    article_text = test_df['content'].tolist()
    article_text = list(map(lambda x: re.sub('\s+',' ', x).strip(), article_text))
    publications = test_df['publication'].tolist()
    write_file(article_text, publications, sys.argv[3])


def write_file(x, y, filename):
    with open(filename, 'w') as f:
        for i in range(len(y)):
            f.write(x[i] + "\t" + y[i] + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s article_dir train_file_name test_file_name train_start_date train_stop_date test_stop_date" % sys.argv[0])
    articles_df = create_dataframe()
    train_start = dt.datetime.strptime(sys.argv[4],'%Y-%m-%d')
    train_stop = dt.datetime.strptime(sys.argv[5],'%Y-%m-%d')
    test_stop = dt.datetime.strptime(sys.argv[6],'%Y-%m-%d')
    write_windowed_data(articles_df, train_start, train_stop, test_stop)
