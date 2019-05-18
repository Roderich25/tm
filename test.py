import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
df = pd.read_csv(r"C:\Users\ravialas\Desktop\corpus.csv", encoding='latin-1')
df = df[['LABEL', 'TEXT']]
df_filter = dict(df.groupby('LABEL').TEXT.count()>49)
df['Bool'] = df['LABEL'].map(df_filter)
df = df[df['Bool'] == True]
df['TEXT'] = [re.sub(r'[^A-Za-z0-9 ]+', ' ', w) for w in df['TEXT']]
df['TEXT'] = [re.sub(' +', ' ', w).lower() for w in df['TEXT']]
#df['TEXT'] = [word_tokenize(entry) for entry in df['TEXT']]
#df['TEXT'] = [word for word in df['TEXT'] if word not in stopwords.words('english')]

df['category_id'] = df['LABEL'].factorize()[0]
category_id_df = df[['LABEL', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'LABEL']].values)

myStopWords = set(stopwords.words('english')) | set(stopwords.words('spanish')) | set(('hi','hello','hola','insurgentes','sur'))

tfidf = TfidfVectorizer(sublinear_tf=True, max_df=1.0, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=myStopWords, max_features=4000)
features = tfidf.fit_transform(df.TEXT).toarray()
labels = df.category_id
features.shape

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(df['TEXT'], df['LABEL'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
predictions_nb = clf.predict(count_vect.transform(X_test))
accuracy_score(predictions_nb, y_test) * 100

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_tfidf, y_train)
predictions_svm = SVM.predict(count_vect.transform(X_test))
accuracy_score(predictions_svm, y_test) * 100