import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string


#creating the dataset
dataset = pd.read_csv("labeled_data.csv")
#print(dataset)
#print(dataset.isnull().sum())
dataset["labels"] = dataset["class"].map({0: "Hate Speech", 1: "Offensive Language",2: "No hate or offensive language"}) 
#print(dataset)
data = dataset[["tweet", "labels"]]
#print(data)

#removing non significant words 
stopwords = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

#cleaning the data
def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('\[.*?\]','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    # Remove the stopwords
    text = [word for word in text.split(' ' ) if word not in stopwords]
    text = " ".join(text)
    # Stemming the text
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean_data)
#print(data)

x = np.array(data["tweet"])
y = np.array(data['labels'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
#print(x_train)

#building ML model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

#Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot = True, fmt="f", cmap="YlGnBu")
plt.show()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
sample = "Let's unite and kill all the people protesting against the government"
sample = clean_data(sample)
print(sample)
data1 = cv.transform([sample]).toarray()
print(data1)
print(dt.predict(data1))