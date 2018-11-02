import pandas as pd
import re
import random

def create_dataframe():
	articles1_df = pd.read_csv(sys.argv[1] + "/articles1.csv")
	articles2_df = pd.read_csv(sys.argv[1] + "/articles2.csv")
	articles3_df = pd.read_csv(sys.argv[1] + "/articles3.csv")
	articles_df = pd.concat([articles1_df, articles2_df, articles3_df])
	return articles_df

def get_features_and_labels(df):
	article_text = articles_df['content'].tolist()
	article_text = list(map(lambda x: re.sub('\s+',' ', x).strip(), article_text))
	publications = articles_df['publication'].tolist()
	z = list(zip(article_text, publications))
	random.shuffle(z)
	article_text, publications = zip(*z)
	return article_text, publications

def generate_train_test(x, y):
	with open(sys.argv[2], 'w') as f:
	    for i in range(len(y[:8000])):
	        f.write(x[i] + "\t" + y[i] + "\n")

	with open(sys.argv[3], 'w') as f:
	    for i in range(len(y[8001:10000])):
	        f.write(x[i] + "\t" + y[i] + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: %s article_dir train_file_name test_file_name" % sys.argv[0])
    articles_df = create_dataframe()
    article_text, publications = get_features_and_labels(articles_df)
    generate_train_test(article_text, publications)