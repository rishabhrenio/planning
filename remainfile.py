iii) UnigramTagger
code:
# Loading Libraries
from nltk.tag import UnigramTagger
from nltk.corpus import treebank

# Training using first 10 tagged sentences of the treebank corpus as data.
# Using data
train_sents = treebank.tagged_sents()[:10]

# Initializing
tagger = UnigramTagger(train_sents)

# Lets see the first sentence 
# (of the treebank corpus) as list   
print(treebank.sents()[0])
print('\n',tagger.tag(treebank.sents()[0]))

#Finding the tagged results after training.
tagger.tag(treebank.sents()[0])

#Overriding the context model
tagger = UnigramTagger(model ={'Pierre': 'NN'})
print('\n',tagger.tag(treebank.sents()[0]))








//d. Compare two nouns
     source code:

import nltk
from nltk.corpus import wordnet

syn1 = wordnet.synsets('football')
syn2 = wordnet.synsets('soccer')

# A word may have multiple synsets, so need to compare each synset of word1 with synset of word2
for s1 in syn1:
    for s2 in syn2:
print("Path similarity of: ")
print(s1, '(', s1.pos(), ')', '[', s1.definition(), ']')
print(s2, '(', s2.pos(), ')', '[', s2.definition(), ']')
print("   is", s1.path_similarity(s2))
print()






//Tokenization using the spaCylibrary

code:
import spacy
nlp = spacy.blank("en")

# Create a string input
str = "I love to study Natural Language Processing in Python"

# Create an instance of document;
# doc object is a container for a sequence of Token objects.
doc = nlp(str)

# Read the words; Print the words
#
words = [word.text for word in doc]
print(words)














//5.	Import NLP Libraries for Indian Languages and perform:
Note: Execute this practical in https://colab.research.google.com/
a) word tokenization in Hindi

Source code:
!pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install inltk

start-!pip install inltk

!pip install tornado==4.5.3     

from inltk.inltk import setup
setup('hi')

from inltk.inltk import tokenize

hindi_text = """प्राकृतिकभाषासीखनाबहुतदिलचस्पहै।"""

# tokenize(input text, language code)
tokenize(hindi_text, "hi")

output
['▁प्राकृतिक', '▁भाषा', '▁सीखना', '▁बहुत', '▁दिलचस्प', '▁है', '।']





b) Generate similar sentences from a given Hindi text input
Source code:
!pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

!pip install inltk

!pip install tornado==4.5.3

from inltk.inltk import setup
setup('hi')

from inltk.inltk import get_similar_sentences

# get similar sentences to the one given in hindi
output = get_similar_sentences('मैंआजबहुतखुशहूं', 5, 'hi')

print(output)

Output:
['मैंआजकलबहुतखुशहूं', 'मैंआजअत्यधिकखुशहूं', 'मैंअभीबहुतखुशहूं', 'मैंवर्तमानबहुतखुशहूं', 'मैंवर्तमानबहुतखुशहूं']




c) Identify the Indian language of a text
Source code:
!pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

!pip install inltk

!pip install tornado==4.5.3

from inltk.inltk import setup
setup('gu')

from inltk.inltk import identify_language
#Identify the Lnaguage of given text
identify_language('બીનાકાપડિયા')



//treeebank
Named Entity recognition with diagram using NLTK corpus – treebank.

Source code:
Note: It runs on Python IDLE

import nltk
nltk.download('treebank')
from nltk.corpus import treebank_chunk
treebank_chunk.tagged_sents()[0]

treebank_chunk.chunked_sents()[0]
treebank_chunk.chunked_sents()[0].draw()




//7.	Finite state automata
a)	Define grammar using nltk. Analyze a sentence using the same.
Code:
import nltk
from nltk import tokenize
grammar1 = nltk.CFG.fromstring("""
	S -> VP
        VP -> VP NP
        NP ->Det  NP
        Det -> 'that'
        NP -> singular Noun
        NP -> 'flight'
        VP -> 'Book'  
	""")
sentence = "Book that flight"

for index in range(len(sentence)):
all_tokens = tokenize.word_tokenize(sentence)
print(all_tokens)

parser = nltk.ChartParser(grammar1)
for tree in parser.parse(all_tokens):
    print(tree)
tree.draw()


//c)	Accept the input string with Regular expression of FA: (a+b)*bba.
Code:
def FA(s):
    size=0
#scan complete string and make sure that it contains only 'a' & 'b'
    for i in s:
        if i=='a' or i=='b':
            size+=1
        else:
            return "Rejected"
#After checking that it contains only 'a' & 'b'
#check it's length it should be 3 atleast
    if size>=3:
#check the last 3 elements
        if s[size-3]=='b':
            if s[size-2]=='b':    
                if s[size-1]=='a':
                    return "Accepted" # if all 4 if true
                return "Rejected" # else of 4th if
            return "Rejected" # else of 3rd if
        return "Rejected" # else of 2nd if
    return "Rejected" # else of 1st if

inputs=['bba', 'ababbba', 'abba','abb', 'baba','bbb','']
for i in inputs:
print(FA(i))


//d)	Implementation of Deductive Chart Parsing using context free grammar and a givensentence.
Source code:
import nltk
from nltk import tokenize
grammar1 = nltk.CFG.fromstring("""
	S -> NP VP
	PP -> P NP
	NP -> Det N | Det N PP | 'I'
	VP -> V NP | VP PP
	Det -> 'a' | 'my'
	N -> 'bird' | 'balcony'
	V -> 'saw'
	P -> 'in'
	""")
sentence = "I saw a bird in my balcony"

for index in range(len(sentence)):
all_tokens = tokenize.word_tokenize(sentence)
print(all_tokens)

# all_tokens = ['I', 'saw', 'a', 'bird', 'in', 'my', 'balcony']
parser = nltk.ChartParser(grammar1)
for tree in parser.parse(all_tokens):
    print(tree)
tree.draw()




//8.	Study PorterStemmer, LancasterStemmer, RegexpStemmer, SnowballStemmer
Study WordNetLemmatizer

Code:
#PorterStemmer
import nltk
from nltk.stem import PorterStemmer
word_stemmer = PorterStemmer()
print(word_stemmer.stem('writing'))







//#RegexpStemmer 
import nltk
from nltk.stem import RegexpStemmer
Reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
print(Reg_stemmer.stem('writing'))







//#WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("word :\tlemma")  
print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos ="a"))








///Implement Naive Bayes classifier
	Code:
	#pip install pandas
#pip install sklearn

import pandas as pd
import numpy as np

sms_data = pd.read_csv("spam.csv", encoding='latin-1')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemming = PorterStemmer()
corpus = []
for i in range (0,len(sms_data)):
    s1 = re.sub('[^a-zA-Z]',repl = ' ',string = sms_data['v2'][i])
    s1.lower()
    s1 = s1.split()
    s1 = [stemming.stem(word) for word in s1 if word not in set(stopwords.words('english'))]
    s1 = ' '.join(s1)
corpus.append(s1)

from sklearn.feature_extraction.text import CountVectorizer
countvectorizer =CountVectorizer()

x = countvectorizer.fit_transform(corpus).toarray()
print(x)

y = sms_data['v1'].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, stratify=y,random_state=2)

#Multinomial Naïve Bayes.
from sklearn.naive_bayes import MultinomialNB
multinomialnb = MultinomialNB()
multinomialnb.fit(x_train,y_train)

# Predicting on test data:

y_pred = multinomialnb.predict(x_test)
print(y_pred)

#Results of our Models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print(classification_report(y_test,y_pred))
print("accuracy_score: ",accuracy_score(y_test,y_pred))

input:
spam.csv file from github
