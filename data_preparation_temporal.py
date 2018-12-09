import pandas as pd
import re
import random
import math
import sys
import datetime as dt

def create_dataframe():
    articles1_df = pd.read_csv(sys.argv[1] + "/articles1.csv")
    stylo1_df = pd.read_csv(sys.argv[1] + "/stylo_data_1.csv")
    all1_df = pd.concat([articles1_df, stylo1_df], axis=1)
    
    articles2_df = pd.read_csv(sys.argv[1] + "/articles2.csv")
    stylo2_df = pd.read_csv(sys.argv[1] + "/stylo_data_2.csv")
    all2_df = pd.concat([articles2_df, stylo2_df], axis=1)
    
    articles3_df = pd.read_csv(sys.argv[1] + "/articles3.csv")
    stylo3_df = pd.read_csv(sys.argv[1] + "/stylo_data_3.csv")
    all3_df = pd.concat([articles3_df, stylo3_df], axis=1)
    
    
    all_df = pd.concat([all1_df, all2_df, all3_df])
    # print(articles_df.groupby('publication').size())
    
    remove_df = pd.read_csv(sys.argv[1] + "/remove_dataset.csv")
    all_df = all_df[~all_df['id'].isin(remove_df['id'])]
    
    return all_df

def write_windowed_data(df, train_start, train_stop, test_stop):
    #get training and test periods
    print('splitting up train and test')
    train_df = df[df['date'].astype('datetime64')>=train_start]
    train_df = train_df[train_df['date'].astype('datetime64')<=test_stop]
    test_df = train_df[train_df['date'].astype('datetime64')>train_stop]
    train_df = train_df[train_df['date'].astype('datetime64')<=train_stop]
    
    #get documents and labels for training data
    print('writing training data to file')
    write_file(train_df, sys.argv[2])
    
    #same for test
    print('writing test data to file')
    write_file(test_df, sys.argv[3])


def write_file(df, filename):
    all_pubs = list(df['publication'].unique())
    article_text = df['content'].tolist()
    article_text = list(map(lambda x: re.sub('\s+',' ', x).strip(), article_text))
    publications = df['publication'].tolist()
    num_sen = df['num_sen'].tolist()
    num_word = df['num_word'].tolist()
    avg_word = df['avg_word_per_sen'].tolist()
    num_entity = df['num_named_entity'].tolist()
    frac_noun = df['frac_words_noun'].tolist()
    frac_verb = df['frac_words_verb'].tolist()
    aps_neg = df['avg_polarity_score_negative'].tolist()
    aps_neut = df['avg_polarity_score_neutral'].tolist()
    aps_pos = df['avg_polarity_score_positive'].tolist()
    aps_cmp = df['avg_polarity_score_compound'].tolist()
    vps_neg = df['var_polarity_score_negative'].tolist()
    vps_neut = df['var_polarity_score_neutral'].tolist()
    vps_pos = df['var_polarity_score_positive'].tolist()
    vps_cmp = df['var_polarity_score_compound'].tolist()
    
    stylo_string = "%d\t%d\t%f\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f"
    with open(filename, 'w') as f:
        for i in range(len(publications)):
            t = (num_sen[i], num_word[i], avg_word[i], num_entity[i], frac_noun[i],
                 frac_verb[i], aps_neg[i], aps_neut[i], aps_pos[i], aps_cmp[i],
                 vps_neg[i], vps_neut[i], vps_pos[i], vps_cmp[i])
            f.write(article_text[i] + "\t" + publications[i] + "\t" + (stylo_string%t) +"\n")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s article_dir train_file_name test_file_name train_start_date train_stop_date test_stop_date" % sys.argv[0])
    articles_df = create_dataframe()
    train_start = dt.datetime.strptime(sys.argv[4],'%Y-%m-%d')
    train_stop = dt.datetime.strptime(sys.argv[5],'%Y-%m-%d')
    test_stop = dt.datetime.strptime(sys.argv[6],'%Y-%m-%d')
    write_windowed_data(articles_df, train_start, train_stop, test_stop)
