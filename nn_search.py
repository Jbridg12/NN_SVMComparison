'''
Josh Bridges
nn_best.py

Using the mushroom dataset, evaluate the accuracy for each model and compare them to
find the best performing.

This file uses a coarse grid search to find the optimal model then
performs a fine grid search based on previous results to find the 
best set of hyperparameters for it.

This code may take upwards of 8 hours to run, the reason is unknown to me other
than there being a lot of iterations. During my testing I never ran the coarse 
and fine grid searches at the same time because of hardware limitations but the 
file does allow that possibility

'''

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm

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
#   General Model               #
# # # # # # # # # # # # # # # # #

def DynamicModel(neuron_one=1, neuron_two=1, activation_one='sigmoid', activation_two='sigmoid'):
    """ A sequential Keras model that has an input layer, one 
        hidden layer with a dymanic number of units, and an output layer."""
    model = Sequential()
    model.add(Dense(neuron_one, input_dim=X.shape[1], activation=activation_one, name='layer_1'))
    model.add(Dense(neuron_two, activation=activation_two, name='layer_2'))
    model.add(Dense(2, activation='sigmoid', name='output_layer'))
     
    # Don't change this!
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model
    
# (4) Evaluation + HyperParameter Search
# Below, we build KerasClassifiers using our model definitions. Use verbose=2 to see
# real-time updates for each epoch.

model = KerasClassifier(
    build_fn=DynamicModel, 
    epochs=200, 
    batch_size=20, 
    verbose=0)
    
    
# # # # # # # # # # # # # # # # # # # # #
# Coarse Grid Search                    #
#   - Broad sweep of hyperparemeters.   #
# # # # # # # # # # # # # # # # # # # # #

# (5) Define a set of unit numbers (i.e. "neurons") and activation functions
# that we want to explore in a coarse-grid search of the hyperparameters.
'''
param_grid = [
    {
        'activation_one': ['linear', 'sigmoid', 'relu', 'tanh'], 
        'activation_two': ['linear', 'sigmoid', 'relu', 'tanh'], 
        'neuron_one': [1, 2, 5, 10, 15, 20, 25, 30],
        'neuron_two': [1, 2, 5, 10, 15, 20, 25, 30]
    }
]

# (6)   Send the Keras model through GridSearchCV, and evaluate the performnce of every option in 
#       param_grid for the "neuron" value.

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X.values, Y)

# (7) Print out a summarization of the results.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''


# # # # # # # # # # # # # # # # # # # # #
# Fine Grid Search                      #
#   - Narrow sweep of hyperparemeters   #
#     for best performing Coarse Search #
#     results                           #
# # # # # # # # # # # # # # # # # # # # #

# (8) Define a set of unit numbers (i.e. "neurons") and activation functions
# that we want to explore in a fine-grid search of the hyperparameters.   
   
param_grid = [
    {
        'activation_one': ['linear'], 
        'activation_two': ['relu'], 
        'neuron_one': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'neuron_two': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    }
]

# (9)   Send the Keras model through GridSearchCV, and evaluate the performance of every option in 
#       param_grid for the "neuron" value.

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X.values, Y)

# (10) Print out a summarization of the results.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
