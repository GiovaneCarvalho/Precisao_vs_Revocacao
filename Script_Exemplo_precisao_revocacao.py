# # 1 - Importando Bibliotecas e recursos
!pip install fastparquet Unidecode seaborn nltk scikit-learn wordcloud transformers datasets xgboost==0.90

import pandas as pd
import fastparquet

import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from string import punctuation
from nltk import tokenize, ngrams
import unidecode
import numpy as np
import re
import os
import shutil

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score, ConfusionMatrixDisplay


#---------- Recursos ----------#
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')
from joblib import dump, load

df = pd.read_parquet(r'.\B2W-Reviews01-tratada-v2.parquet')
df.head()

def tratar(df, coluna):

  palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
  pontuacao = []
  for ponto in punctuation:
    pontuacao.append(ponto)
  pontuacao_stopwords = pontuacao + palavras_irrelevantes
  token_pontuacao = tokenize.WordPunctTokenizer()
  sem_acentos = [unidecode.unidecode(texto) for texto in df[coluna]]
  stopwords_sem_acentos = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]
  stemmer = nltk.RSLPStemmer()

  df['texto_tratado'] = sem_acentos

  frase_processada = []

  for texto in df['texto_tratado']:

    nova_frase = []
    texto = texto.lower()
    texto = texto.replace('http','')
    texto = texto.replace('...','')
    texto = texto.replace('://','')
    texto = texto.replace('..','')
    texto = re.sub('\d','', texto)
    palavras_texto = token_pontuacao.tokenize(texto)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))

  df['texto_tratado'] = frase_processada

tratar(df,'review_text')

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()

var='polarity'
X = df['texto_tratado']
y = df[var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_df = 0.4, ngram_range = (1,3), max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
classifier = LogisticRegression(max_iter = 500, class_weight='balanced')
classifier.fit(X_train_tfidf, y_train)
X_test_tfidf = vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_tfidf)
b_accuracy = balanced_accuracy_score(y_test, y_pred)
print('#------------------------------------------------------------#')
print("Balanced Accuracy:", b_accuracy)
print('#------------------------------------------------------------#')
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print('#------------------------------------------------------------#')
print(classification_report(y_test, y_pred))
print('#------------------------------------------------------------#')
c_r = classification_report(y_test, y_pred)

cm_log = confusion_matrix(y_pred,y_test)
plot_confusion_matrix(cm_log)

y_prob = classifier.predict_proba(X_test_tfidf)

ranges = list(np.arange(0, 1, .005))

from sklearn.metrics import  precision_score, recall_score, precision_recall_curve

y_test = np.array([1 if x =='Negativo' else 0 for x in y_test])

y_test, y_pred_sim, y_prob[:,1]

revocacao = []
precisao = []


for i in ranges:
    y_pred_sim = np.array([1 if x > i else 0 for x in list(y_prob[:,0])])
    revocacao.append(recall_score(y_test, y_pred_sim))
    precisao.append(precision_score(y_test, y_pred_sim))

plt.figure(figsize = (12,9))
plt.plot(ranges, revocacao)
plt.plot(ranges, precisao)
plt.legend(['Revocação', 'Precisão'])
plt.show()


