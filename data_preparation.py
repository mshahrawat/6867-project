import pandas as pd
import re
import random

articles1_df = pd.read_csv("article_data/articles1.csv")
articles2_df = pd.read_csv("article_data/articles2.csv")
articles3_df = pd.read_csv("article_data/articles3.csv")

articles_df = pd.concat([articles1_df, articles2_df, articles3_df])
article_text = articles_df['content'].tolist()
article_text = list(map(lambda x: re.sub('\s+',' ', x).strip(), article_text))
publications = articles_df['publication'].tolist()
z = list(zip(article_text, publications))
random.shuffle(z)
article_text, publications = zip(*z)

with open('data/data_8000_train.txt', 'w') as f:
    for i in range(len(publications[:8000])):
        f.write(article_text[i] + "\t" + publications[i] + "\n")

with open('data/data_8000_test.txt', 'w') as f:
    for i in range(len(publications[8001:10000])):
        f.write(article_text[i] + "\t" + publications[i] + "\n")