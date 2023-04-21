import pandas as pd
import numpy as np 
import re
from sklearn.feature_extraction import _stop_words
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer


df_train = pd.read_csv('data/imdb_data.csv')
df_test = pd.read_csv('data/imdb_data_fake.csv')

text_col = df_train['review']
text_col_test = df_test['review']

text_col = text_col.apply(lambda x: x.lower())
text_col_test = text_col_test.apply(lambda x: x.lower())

text_col = text_col.apply(lambda x: x.replace('<br />', ' '))
text_col_test = text_col_test.apply(lambda x: x.replace('<br />', ' '))
text_col = text_col.apply(lambda x: x.replace('<br/>', ' '))
text_col_test = text_col_test.apply(lambda x: x.replace('<br/>', ' '))

text_col = text_col.apply(lambda x: re.sub(r'[^\w\s]','',x))
text_col_test = text_col_test.apply(lambda x: re.sub(r'[^\w\s]','',str(x)))

text_col = text_col.apply(lambda x:[xx.strip() for xx in x.split() if xx not in _stop_words.ENGLISH_STOP_WORDS])
text_col_test = text_col_test.apply(lambda x:[xx.strip() for xx in x.split() if xx not in _stop_words.ENGLISH_STOP_WORDS])

le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['sentiment'])
df_test['label'] = le.fit_transform(df_test['sentiment'])

vectorizer = HashingVectorizer(n_features=1000)
text_col_sparse = vectorizer.fit_transform(text_col.apply(lambda x: ' '.join(x)))
text_col_test_sparse = vectorizer.fit_transform(text_col_test.apply(lambda x: ' '.join(x)))

train_clean = pd.concat([pd.DataFrame(text_col_sparse.toarray()), df_train['label']], axis=1)
test_clean = pd.concat([pd.DataFrame(text_col_test_sparse.toarray()), df_test['label']], axis=1)

train_clean.to_csv('./data/train_data.csv', index=False)
test_clean.to_csv('./data/test_data.csv', index=False)