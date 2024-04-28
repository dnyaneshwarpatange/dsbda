import nltk 
import re

nltk.download('punkt') 
nltk.download('stopwords') 
nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')


file_path = './file.txt'
file = open(file_path, 'r')
# Read the contents of the file
file_contents = file.read()
text = file_contents
# Close the file
file.close()
# Print the contents of the file
print(file_contents)

# sentence tokenization
from nltk.tokenize import sent_tokenize
tokenized_text = sent_tokenize(text)
print("tokenized text>>>> ",tokenized_text)


# word tokenization 
from nltk.tokenize import word_tokenize
tokenized_word = word_tokenize(text)
print("tokenized word>>>> ",tokenized_word)

# print stop words of english
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
print('stop_words>>>> ',stop_words)

text= "How to remove stop words with NLTK library in Python?" 
text= re.sub('[^a-zA-Z]', ' ',text)
tokens = word_tokenize(text.lower()) 
filtered_text=[]
for w in tokens:
    if w not in stop_words:
        filtered_text.append(w) 
print ("Tokenized Sentence:",tokens) 
print ("Filtered Sentence:",filtered_text)


from nltk.stem import PorterStemmer 
e_words= ["wait", "waiting", "waited", "waits"] 
ps =PorterStemmer()
for w in e_words:
    rootWord=ps.stem(w) 
print(rootWord)

from nltk.stem import WordNetLemmatizer 
wordnet_lemmatizer = WordNetLemmatizer()
 
text = "studies studying cries cry" 
tokenization = nltk.word_tokenize(text) 
for w in tokenization:
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))

import nltk
from nltk.tokenize import word_tokenize 
data="The pink sweater fit her perfectly" 
words=word_tokenize(data)

for word in words:
    print(nltk.pos_tag([word]))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

d0 = 'Jupiter is the largest planet'
d1 = 'Mars is the fourth planet from the Sun' 
string = [d0, d1]
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(string) 
print('\nWord Indices: ') 
print(tfidf.vocabulary_) 
print('\ntfidf values: ') 
print(result)
