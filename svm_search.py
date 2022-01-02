'''
Josh Bridges
svm_search.py

This file uses a coarse grid search followed by a fine grid search of the hyperparameters
to evaluate each type of SVM and identify the best performing one on the mushroom dataset.

Each SVM-hyperparameter setup is evaluated over 7-fold cross evaluation and the accuracy
for the model is averaged over the folds. The best performing kernel from coarse grid search,
identified from prior runs, is then evaluated in a fine grid search with a more in depth 
search of the hyperparameters for the best kernel.

'''
import numpy as np
import pandas as pd
from statistics import mean
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# By default, Sklearn forces warnings into your terminal.
# Here, we're writing a dummy function that overwrites the function
# that prints out numerical warnings.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

filename = "mushrooms.csv"
df = pd.read_csv(filename)
df = df.replace('?', pd.NaT)    # Replace the '?' characters with nan to allow easier removal
df = df.dropna()                # Drop all rows that have any unknown data

X = df.iloc[:,1:]
X = pd.get_dummies(X).astype(int)  #one hot encoding

Y = df[['class']]
Y = LabelEncoder().fit_transform(Y.values.ravel())  # Encode the results into a binary representation


# # # # # # # # # # # # # # # # # # # # #
# Coarse Grid Search                    #
#   - Broad sweep of hyperparemeters.   #
# # # # # # # # # # # # # # # # # # # # #

# Set the parameters for coarse search
tuned_parameters = [
    {
        'kernel': ['linear'], 
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['poly'], 
        'degree': [2, 3, 4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['rbf'], 
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    }
]

'''
# # # # # # # # # # # # # # # # # # # # #
# Fine Grid Search                      #
#   - Narrow sweep of hyperparemeters   #
#     for best performing Coarse Search #
#     results                           #
# # # # # # # # # # # # # # # # # # # # #

# Set the parameters for fine search
tuned_parameters = [
    {
        'kernel': ['poly'], 
        'degree': [1, 2, 3, 4, 5, 6],
        'C': [0.1, 1, 5, 7, 10, 100, 1000, 10000]
    }
]
'''
scores = ['accuracy']   # Evaluate best model based on the accuracy metric

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV( 
        SVC(), 
        tuned_parameters, 
        scoring='%s' % score, 
        cv=7,                   # Perfom a 7-fold Grid Search
        n_jobs = -2)
        
    clf.fit(X.values, Y)        # Train the model on the test data

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    


