import pandas as pd
import re
import random
import math
import sys

def create_dataframe():
	articles1_df = pd.read_csv(sys.argv[1] + "/articles1.csv")
	articles2_df = pd.read_csv(sys.argv[1] + "/articles2.csv")
	articles3_df = pd.read_csv(sys.argv[1] + "/articles3.csv")
	articles_df = pd.concat([articles1_df, articles2_df, articles3_df])
	# print(articles_df.groupby('publication').size())
	return articles_df

def get_features_and_labels(df):
	all_pubs = list(df['publication'].unique())
	article_text = df['content'].tolist()
	article_text = list(map(lambda x: re.sub('\s+',' ', x).strip(), article_text))
	publications = df['publication'].tolist()
	z = list(zip(article_text, publications))
	sampled = []
	for publication in all_pubs:
		only_pub = list(filter(lambda x: x[1] == publication, z))
		sampled.extend(random.sample(only_pub, 2000))
	random.shuffle(sampled)
	article_text, publications = zip(*sampled)
	return article_text, publications

def generate_train_test(x, y):
	num_samples = len(y) #30,000
	train_test_split = int(sys.argv[4])
	val = False
	if val:
		pass # TODO 
	else :
		split = math.ceil(num_samples * train_test_split / 100.)
		with open(sys.argv[2], 'w') as f:
		    for i in range(len(y[:split])):
		        f.write(x[i] + "\t" + y[i] + "\n")
		with open(sys.argv[3], 'w') as f:
		    for i in range(len(y[split:])):
		        f.write(x[i+split] + "\t" + y[i+split] + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s article_dir train_file_name test_file_name train_test_split" % sys.argv[0])
    articles_df = create_dataframe()
    article_text, publications = get_features_and_labels(articles_df)
    generate_train_test(article_text, publications)

