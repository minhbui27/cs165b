# Starter code for CS 165B MP3
import math
import random
import numpy as np
import pandas as pd

from typing import List
def perceptron_learning_algorithm(training_data: pd.DataFrame, learning_rate = 1e-3, epochs = 500):
    weights = np.random.rand(6)
    errors = 1
    counter = 0
    while(errors != 0 and counter != epochs):
        errors = 0
        counter += 1
        for idx, row in training_data.iterrows():
            label = row['target']
            row = np.append(row.to_numpy()[:5],1)
            if(np.dot(row,weights) >= 0 and label == 0):
                errors += 1
                weights = weights - learning_rate*row
            elif(np.dot(row,weights) < 0 and label == 1):
                errors += 1
                weights = weights + learning_rate*row
        # print(errors)
    return weights


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: pd.DataFrame
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    weights = perceptron_learning_algorithm(training_data,1e-3,100)
    output = np.zeros(len(testing_data['x1']))
    for idx, row in testing_data.iterrows():
        row = np.append(row.to_numpy(),1)
        output[idx] = 1 if np.dot(row,weights) >= 0 else 0
    return output

    #TODO implement your model and return the prediction

if __name__ == '__main__':
    # load data
    training = pd.read_csv('data/train.csv')
    testing = pd.read_csv('data/dev.csv')
    target_label = testing['target']
    testing.drop('target', axis=1, inplace=True)

    # run training and testing
    prediction = run_train_test(training, testing)
    # check accuracy
    target_label = target_label.values
    print("Dev Accuracy: ", np.sum(prediction == target_label) / len(target_label))
    


    


