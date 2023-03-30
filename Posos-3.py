from __future__ import print_function
import numpy as np
from numpy import *
import csv
from sklearn import datasets, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
#from hmmlearn import hmm
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# from nltk.stem import *
# from nltk.stem.snowball import SnowballStemmer

vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
vectorizer

C = []
# Stemmer = SnowballStemmer("french")

with open('input_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[1] == 'question': continue
        C.append(row[1])

# with open('input_test.csv') as csvfile:
#    reader = csv.reader(csvfile, delimiter=';')
#    for row in reader:
#	    if row[1] == 'question': continue
#	    C.append(row[1])

y = np.array([])
with open('output_train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if row[1] != 'intention': y = np.concatenate([y, [float(row[1])]])

X_all = vectorizer.fit_transform(C)
#print(X_all.shape)
X = X_all[0:8028]

select = SelectKBest(k=6000)
X_new = select.fit_transform(X, y)
#print(X_new.shape)
f = select.get_support([indices])

X = X_new


# clf = LogisticRegression(solver='sag', max_iter=1000, random_state=42,multi_class='multinomial').fit(X_new, y)
# print(clf.coef_)
# print(clf.predict(X_all[8028:10063,f]))
# pred = clf.predict(X_all[0:8028,f])
# print(pred)
# print(Y)
# print('Features: %s' % vectorizer.get_feature_names())
# print(X)

def score_function(y_true, y_pred):
    score = 0
    length1 = y_true.shape[0]
    for i in range(length1):
        if y_pred[i] == y_true[i]:
            score += 1
    return float(score) / float(length1)


# print(score_function(y,pred))

cv = KFold(n=len(y), n_folds=5)
results = []
# clf = LogisticRegression(solver='newton-cg', max_iter=10000, random_state=42,multi_class='multinomial')

for training_set, test_set in cv:
    X_train = X[training_set]/1.32
    y_train = y[training_set]
    X_test = X[test_set]/1.32
    #scaler = StandardScaler(copy=True, with_mean=False, with_std=True)
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    y_test = y[test_set]
    #X_test = scaler.transform(X_test)

    #clf = LinearSVC(random_state=0,C=.095)
    #clf.fit(X_train, y_train)
    #y_prediction=clf.predict(X_test)
    
    model = Sequential()
    model.add(Dense(64, input_dim=n_features, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.fit(X_train, y_train)
    #model.evaluate(X_train, y_train)
    y_prediction = model.predict(X_test)
    
    #mlp = linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
    #normalize=False, positive=False, precompute=False, random_state=None,
    #selection='cyclic', tol=0.0001, warm_start=False)
    
    
    #mlp.fit(X_train,y_train)
    

    #lass = linear_model.LogisticRegression(penalty="l1", dual=False, tol=0.0001, C=1.0, 
    #                                       fit_intercept=True, intercept_scaling=0, class_weight=None, 
    """
    random_state=None, solver="sag", max_iter=1000, 
    multi_class="ovr", verbose=0, warm_start=False, n_jobs=1)
    """
    #lass.fit(X_train, y_train)
    #mlp = hmm.GaussianHMM(n_components=50)
    #mlp.fit(X_test.toarray())
    #print(mlp.transmat_)
    
    #pca = PCA(n_components=2000)
    #pca.fit(X_train.toarray())
    #X_train=pca.transform(X_train.toarray())
    
    #pca.fit(X_test.toarray())
    #X_test=pca.transform(X_test.toarray())
    
    #MNB=MultinomialNB(alpha=1, fit_prior=True, class_prior=None)
    #MNB.fit(X_train,y_train)
    #y_prediction = MNB.predict(X_test)
    #print(y_prediction)
    
    #mlp = MLPClassifier(activation='relu', alpha=.01, batch_size='auto',
    #                                  beta_1=0.9, beta_2=0.999, early_stopping=True,
     #                                 epsilon=1e-08, hidden_layer_sizes=(200,200,200), learning_rate='invscaling',
      #                                learning_rate_init=0.001, max_iter=50, momentum=0.9,
       #                               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
        #                              solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
         #                             warm_start=False)


    #clf = RandomForestClassifier(n_estimators=400)
    #clf.fit(X_train, y_train)
    #y_prediction = clf.predict(X_test)
    #ada=AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.3, algorithm="SAMME", random_state=None)
    #ada.fit(X_train, y_train, sample_weight=None)
    #y_prediction=ada.predict(X_test)
    #y_prediction=lass.predict(X_test)"""
    print(y_prediction)
    result = np.sum(y_test == y_prediction) * 1. / len(y_test)
    results.append(result)
    print("prediction accuracy:", result)

print("overall prediction accuracy:", np.mean(results))
