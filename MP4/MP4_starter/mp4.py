# Starter code for CS 165B MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import sklearn
from sklearn.neural_network import MLPClassifier

np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """

    # Separating the training labels and training data to use for training
    training_labels = training_data['target']
    training_data.drop('target', axis=1, inplace=True)

    # These are the feature with large values I will normalize
    continuous_features = ['DAYS_EMPLOYED','DAYS_BIRTH','AMT_INCOME_TOTAL']
    for feature in continuous_features:
        scaler = sklearn.preprocessing.StandardScaler()
        training_data[feature] = scaler.fit_transform(training_data[feature].values.reshape(-1,1))
        testing_data[feature] = scaler.fit_transform(testing_data[feature].values.reshape(-1,1))

    # Categorical col if the first value in the column is a string
    categorical_cols = []
    for col in training_data:
        if(type(training_data[col][0]) == str):
            categorical_cols.extend([col])

    # Then we need to perform one hot transformation on training and testing data
    training_data = pd.get_dummies(training_data,columns=categorical_cols)
    testing_data = pd.get_dummies(testing_data,columns=categorical_cols)
    
    # For all columns not in training data that is in testing data, remove this from the testing set due to irrelevance
    for col in testing_data:
        if col not in training_data:
            testing_data = testing_data.drop(col,axis=1) 
        
    # For all columns in training data that isn't in testing data, add a column of zeros with this labl to match dimensionality
    # At the same time record the ordering of the training dataset 
    training_ordering = []
    for col in training_data:
        training_ordering.extend([col])
        if col not in testing_data:
            testing_data[col] = 0

    # Reorder the testing set to match the training set
    testing_data = testing_data.reindex(columns=training_ordering)

    # Creating the classifier
    clf = MLPClassifier(verbose=True,hidden_layer_sizes=(128,64,64), max_iter=1000, learning_rate_init= 1e-4, learning_rate='constant')

    # Fitting the training data to the labels
    clf.fit(training_data,training_labels)

    # Returning the prediction
    return clf.predict(testing_data)  


if __name__ == '__main__':

    training = pd.read_csv('data/train.csv')
    development = pd.read_csv('data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)
