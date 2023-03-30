from __future__ import print_function
import numpy as np
from numpy import *
import csv
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold	

#vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
vectorizer = CountVectorizer(analyzer='char',max_df=.7,min_df=3, ngram_range=(4, 4))

#from nltk.stem import *
#from nltk.stem.snowball import SnowballStemmer

C = []
#Stemmer = SnowballStemmer("french")

with open('input_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[1] == 'question': continue
        C.append(row[1])

with open('input_test.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
	    if row[1] == 'question': continue
	    C.append(row[1])

y = np.array([])
with open('output_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
    	if row[1] != 'intention': y = np.concatenate([y,[float(row[1])]])

ny = len(y)
X_all = vectorizer.fit_transform(C)/1.32
(nx,_) = shape(X_all)
#print(X_all.shape)
X = X_all[0:ny]

select = SelectKBest(k=6000)
X_new = select.fit_transform(X, y)
#print(X_new.shape)
f = select.get_support([indices])
#print(f)

clf = LinearSVC(random_state=0,C=.095).fit(X_new, y)
#clf = LogisticRegression(solver='newton-cg', max_iter=10000, random_state=42,multi_class='multinomial').fit(X_new, y)
#print(clf.coef_)
pred = clf.predict(X_all[ny:nx,f])

output = np.zeros((nx-ny, 2))
for i in range(ny,nx):
	output[i-ny,0] = int(i)
	output[i-ny,1] = int(pred[i-ny])

#np.savetxt("output_test.csv", output, delimiter=";")
pd.set_option('precision', 0)
df = pd.DataFrame(output,columns=['ID', 'intention'])
df.astype(int)
df.round()
df = df.fillna(0)
df = df.astype(int)
df.to_csv("output_test.csv",index=False, sep=";",line_terminator="\r\n")
print(output)
pred_self = clf.predict(X_all[0:8028,f])
#print(pred)
#print(Y)
#print('Features: %s' % vectorizer.get_feature_names())
#print(X)

def score_function(y_true, y_pred):
    score = 0
    length1 = y_true.shape[0]
    for i in range(length1):
        if y_pred[i] == y_true[i]:
            score += 1
    return float(score)/float(length1)

print(score_function(y,pred_self))

