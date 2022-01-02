'''
Josh Bridges
svm_best.py

Using the best model found in svm_search, this file analyzes the models
performance on the mushroom dataset with three metrics: Precision, Recall, and Accuracy.

This file uses multiple different amounts of k folds ranging from 7-13, based on 
data from svm_search.py, to evaluate the performance of the best model.

The three metrics are displayed to the console for each value of k and 
a Precision-Recall plot is created from the metrics for each fold.

'''
import pandas as pd
import matplotlib.pyplot as pl
from statistics import mean
from sklearn import svm
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
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

scoring = {'acc' : 'accuracy', 'precision' : 'precision_macro', 'recall' : 'recall_macro'} # Define the scoring metrics to evaluate

# Store results from each iteration in lists for plotting
pre = []
rec = []

poly_svc_three = svm.SVC(kernel='poly', C=1, degree=3, probability=True)

for k in range(7, 13):
    predicted_poly_three = cross_validate(poly_svc_three, X.values, Y, scoring=scoring, cv=k)
    
    
    # Collect results in the proper lists and display the values for each K vlaue to console
    print('- %d-Fold Cross Validation -' % k)
    print("Accuracy: SVM + Poly (D=3)\t-> " + str(mean(predicted_poly_three['test_acc'])))
    
    print("Precision: SVM + Poly (D=3)\t-> " + str(mean(predicted_poly_three['test_precision'])))
    pre.append(mean(predicted_poly_three['test_precision']))
    
    print("Recall: SVM + Poly (D=3)\t-> " + str(mean(predicted_poly_three['test_recall'])))
    rec.append(mean(predicted_poly_three['test_recall']))
    print()


# Use a new split to generate a Precision-Recall Curve with the model
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2, random_state=1)
poly_svc_three.fit(X_train, Y_train)

display = PrecisionRecallDisplay.from_estimator(
    poly_svc_three, X_test, Y_test, name="PolySVC"
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
pl.show()
