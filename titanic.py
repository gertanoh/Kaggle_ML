import numpy as np
from csv import reader
import matplotlib.pyplot as plt
import random
from random import seed
import pandas as pd
import csv
from sklearn import datasets, svm, preprocessing, metrics
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import math
"""
Binary classification
on Titanic dataset from Kaggle
"""


seed(1)
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        # skip header
        #next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def prepare(dataset, train=True):
    
   
    # no need for passenger id
    dataset = dataset.drop(['PassengerId'], axis=1)
    # remove name and keep titles and use titles fo fill age na
    dataset['Name'] = dataset['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    titles = dataset['Name'].unique()
    dataset['Age'].fillna(-1, inplace=True)
    # there is only one women with Ms, i would group with Miss
    medians = dict()
    for title in titles:
        median = dataset.Age[(dataset['Age'] != -1) & (dataset['Name'] == title)].median()
        if math.isnan(median):
            median = medians['Miss']
            print median
        medians[title] = median 
    for index, row in dataset.iterrows():
        if row['Age'] == -1.0:
            dataset.at[index, 'Age'] = medians[row['Name']]
            
    if train == False:
        dataset['Fare'].fillna(-1, inplace=True)
        for title in titles:
            median = dataset.Age[(dataset["Fare"] != -1) & (dataset['Name'] == title)].median()
            medians[title] = median   
        for index, row in dataset.iterrows():
            if row['Fare'] == -1:
                dataset.loc[index, 'Fare'] = medians[row['Name']]
    # influence of name into survived
    replacement = {
    'Don': 0,
    'Dona': 0,
    'Rev': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Mr': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Master': 4,
    'Miss': 5,
    'Mrs': 6,
    'Mme': 7,
    'Ms': 7,
    'Mlle': 7,
    'Sir': 7,
    'Lady': 7,
    'the Countess': 7
    }
    dataset['Name'] = dataset['Name'].apply(lambda x: replacement.get(x))
    dataset['Name'] = StandardScaler().fit_transform(dataset['Name'].values.reshape(-1, 1))
    
    #processed = dataset.copy()
    le = preprocessing.LabelEncoder()
    # encode sex and embarked into values 
    dataset.Sex = le.fit_transform(dataset.Sex)
    dataset.Embarked = le.fit_transform(dataset.Embarked)
    
    # drop cabin
    dataset.drop(['Cabin', 'Ticket'], axis = 1, inplace = True)
    dataset_null = dataset.isnull().unstack()
   
    if train:
        X = dataset.drop(['Survived'], axis=1).values
        y = dataset['Survived'].values
        return X, y
  
    
    return dataset
    



# test classifier
# use of cross validation technique
# number of folds
def evaluate_algorithm(X , y, classifier, n_folds=5):
    scores = cross_val_score(classifier, X, y, cv=n_folds)
    #print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))
    return scores.mean() * 100
        


#check best classifiers
def compare_classifiers(X, y, classifiers, n_folds=10, n_iter=20):
    scores = list()
    for classifier in classifiers:
        score_tmp = list()
        for iter in range(n_iter):
            score_tmp.append(evaluate_algorithm(X, y, classifier, n_folds))
        
        scores.append(sum(score_tmp) / n_iter )
                      
    return scores


def check_bagging(X, y, n_iter=20, n_folds=10):
    samples = [0.5,0.6,0.7,0.8]
    scores = list()
    for max_sample in samples:
        score_tmp = list()
        for i in range(n_iter):
            b_tree = DecisionTreeClassifier(max_depth=10)
            bagging_tree = BaggingClassifier(b_tree, max_samples=max_sample)
            tmp = cross_val_score(bagging_tree, X, y, cv=n_folds).mean()
            score_tmp.append(tmp)
        
        scores.append(sum(score_tmp) / len(score_tmp) )
    print scores
    print " correct percentage of sample ", samples[scores.index(max(scores))]
    return samples[scores.index(max(scores))]

def estimate_number_trees(X, y, n_iter=20, n_folds=5):
    n_trees = [2,3,4,5,6,7,8,9,10]
    scores = list()
    for tree in n_trees:
        score_tmp = list()
        for i in range(n_iter):
            b_tree = DecisionTreeClassifier(max_depth=10)
            bagging_tree = BaggingClassifier(b_tree, max_samples=0.5, n_estimators=tree)
            tmp = cross_val_score(bagging_tree, X, y, cv=n_folds).mean()
            score_tmp.append(tmp)
        
        scores.append(sum(score_tmp) / len(score_tmp) )
    #print scores
    #print " correct percentage of sample ", n_trees[scores.index(max(scores))]
    return n_trees[scores.index(max(scores))]
    
    
    
    
    
train_file = "train.csv"
test_file = "test.csv"

train = pd.read_csv(train_file,index_col=None, na_values=['NA'])
test = pd.read_csv(test_file,index_col=None, na_values=['NA'])

# Preparing the data
X , y = prepare(train)

# Test
tree = DecisionTreeClassifier()
b_tree = BaggingClassifier(n_estimators=100)
r_forest = RandomForestClassifier(n_estimators=100)


classifiers = list()
classifiers.append(r_forest)
classifiers.append(b_tree)
scores = compare_classifiers(X, y, classifiers)
print('Scores: %s' % scores)

# final choice is bagging
#X_test = prepare(test, train=False)
#print "len of test ", len(X_test)



#b_tree.fit(X, y)
#out = b_tree.predict(X_test)
##print out
#passengerId_row = test['PassengerId'].values
##print len(passengerId_row)
##print len(out)
#rows = zip(passengerId_row, out)
#with open('submission.csv', 'wb') as file:
    #writer = csv.writer(file)
    #writer.writerow(["PassengerId", "Survived"])
    #for row in rows:
        #writer.writerow(row)
    
    



                

