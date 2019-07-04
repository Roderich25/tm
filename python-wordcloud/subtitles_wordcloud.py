import re
import string
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

pattern1 = r'.*-->.*'
pattern2 = r'\d{0,3}\n'

with open('s1e1.txt', 'r', encoding="utf-8") as f:
    contents = f.read()
    contents = re.sub(pattern1, ' ', contents)
    contents = re.sub(pattern2, ' ', contents)
    contents = re.sub('\W+', ' ', contents)
translator = str.maketrans(' ', ' ', string.punctuation)
contents = contents.translate(translator)
contents = contents.lower().split(' ')
contents = list(filter(lambda sub: sub != '', contents))
stop_words = stopwords.words('german')
contents = [w for w in contents if not w in stop_words]
c = Counter(contents)  # c returns a Dictionary
print(c.most_common(100))  # most_common method returns a list of tuples
wc = WordCloud(background_color="white", width=1000, height=1000, max_words=100, relative_scaling=0.5,
               normalize_plurals=False
               ).generate_from_frequencies(c)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.savefig('s1e1.png')
plt.show()
