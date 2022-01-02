'''
Josh Bridges
nn_best.py

This file uses the best found model from nn_search.py to evaluate more
metrics for analysis. 

Using the mushroom dataset, this program outputs the Accuracy, Precision, and Recall 
averaged over k-folds to the console. It also creates a Precision-Recall plot of
the data for each fold.

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from statistics import stdev, mean
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.metrics import Precision, Recall
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold

# (0) Hide as many warnings as possible!
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
tf.compat.v1.disable_eager_execution()

# (1) Read in the Mushroom dataset.
filename = "mushrooms.csv"
df = pd.read_csv(filename)
df = df.replace('?', pd.NaT) # Replace the '?' characters with nan characters so they can easily be dropped
df = df.dropna()             # Remove all rows with any unknowns   
                      
# (2a) Use one-hot encoding technique to "binarize" the feature set

X = df.iloc[:,1:]
X = pd.get_dummies(X).astype(int)  #one hot encoding

# (2b) Create an encoder that "binarizes" target labels.
Y = df[['class']]
Y = LabelEncoder().fit_transform(Y.values.ravel())


# (3) Build Keras model.
# # # # # # # # # # # # # # # # # 
#   Model 1:   An Optimal Model #
# # # # # # # # # # # # # # # # #
def OptimalModel():
    """ A sequential Keras model that has an input layer, one 
        hidden layer, and an output layer."""
    model = Sequential()    
    model.add(Dense(15, input_dim=X.shape[1], activation='linear', name='layer_1'))
    model.add(Dense(30, activation='relu', name='layer_2'))
    model.add(Dense(2, activation='sigmoid', name='output_layer'))
     
     
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics= ['accuracy', Precision(), Recall()]) # The model evaluates precision and recall on top of accuracy
    return model


# (4) Model Evaluations
# Below, we build KerasClassifiers using our model definitions. Use verbose=2 to see
# real-time updates for each epoch.

# Build the model to excute 200 epochs
estimator = KerasClassifier(
        build_fn=OptimalModel,
        epochs=200, batch_size=20,
        verbose=2)

print("- - - - - - - - - - - - - ")

X_train, X_test, Y_train, Y_test = train_test_split( X.values, Y, test_size=0.1, random_state=12)

estimator.fit(X_train, Y_train)    # Train the model  
y_pred = estimator.predict(X_test)    # Get the predictions of the test for metrics

print("(Accuracy) " + "Performance: %.2f%%" % (accuracy_score(Y_test, y_pred)*100))   # Display the three metrics to the console
print("(Recall) " + "Performance: %.2f%%" % (recall_score(Y_test, y_pred)*100))
print("(Precision) " + "Performance: %.2f%%" % (precision_score(Y_test, y_pred)*100))


# (5) Plotting code for PnR curve
# Use sklearn methods to generate a precision and recall curve
# and display it using matplotlib
precision, recall, _ = precision_recall_curve(Y_test, y_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
pl.show()
