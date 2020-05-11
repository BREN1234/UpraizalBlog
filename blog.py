import requests
import string
import nltk
import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import  Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
responsePage= requests.get("https://www.upraizal.com/top-7-elements-ideal-employee-performance-appraisal")

stringParagraph=""
if responsePage.status_code == 200:
    htmlTags= responsePage.text
    text= BeautifulSoup(htmlTags)
    allParagraph= text.find_all('div',{'class':'entry-content'})
    for paragraph in allParagraph:
        stringParagraph = stringParagraph + paragraph.text.strip()
    stringParagraph1= re.sub(r'[^\w\s]|\n|\t',' ',stringParagraph)
    stringParagraph1= " ".join(stringParagraph1.split())
    #term vector initialized
    tfidf= TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidfTrained= tfidf.fit_transform([stringParagraph1])
    # remove stopwords
    words = [word for word in stringParagraph1.lower().strip().split() if word not in stop_words]
    # count word frequency, sort and return just 10
    counter = Counter(words)
    most_common = counter.most_common(10)

    # Get the words /term in Document
    wordsInDoc = tfidf.get_feature_names()
    # sum tfidf frequency of each term through documents
    sums = tfidfTrained.sum(axis=0)

    # term to its sums frequency
    data = []
    for col, word in enumerate(wordsInDoc):
        data.append( (word, sums[0,col] ))

    df = pd.DataFrame(data, columns=['term/words','tfidf'])
    print(df.sort_values('tfidf', ascending=False).head(10))

    print(most_common)
