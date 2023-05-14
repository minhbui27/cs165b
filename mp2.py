# Starter code for CS 165B MP2
# Decision Truee


import os
import sys
import json
import numpy as np
import pandas as pd

from typing import List

# define the Node structure for Decision Tree
class Node:
    def __init__(self) -> None:
        self.left = None            # left child, a Node object
        self.right = None           # right child, a Node object
        self.split_feature = None   # the feature to be split on, a string
        self.split_value = None     # the threshould value of the feature to be split on, a float
        self.is_leaf = False        # whether the node is a leaf node, a boolean
        self.prediction = None      # for leaf node, the class label, a int
        self.ig = None              # information gain for current split, a float
        self.depth = None           # depth of the node in the tree, root will be 0, a int

class DecisionTree():
    """Decision Tree Classifier."""
    def __init__(self, max_depth:int, min_samples_split:int, min_information_gain:float =1e-5) -> None:
        """
            initialize the decision tree.
        Args:
            max_depth: maximum tree depth to stop splitting. 
            min_samples_split: minimum number of data to make a split. If smaller than this, stop splitting. Typcial values: 2, 5, 10, 20, etc.
            min_information_gain: minimum ig gain to consider a split to be valid.
        """
        self.root = None                                    # the root node of the tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_information_gain  = min_information_gain

    def fit(self, training_data: pd.DataFrame, training_label: pd.Series):
        '''
            Fit a Decission Tree based on data
            Args:
                training_data: Data to be used to train the Decission Tree
                training_label: label of the training data
            Returns:
                root node of the Decission Tree
        '''
        self.root = self.GrowTree(training_data, training_label, counter = 0)
        return self.root
  
    def GrowTree(self, data: pd.DataFrame, label: pd.Series, counter: int=0):
        '''
            Conducts the split feature process recursively.
            Based on the given data and label, it will find the best feature and value to split the data and reture the node.
            Specifically:
                1. Check the depth and sample conditions
                2. Find the best feature and value to split the data by BestSplit() function
                3. Check the IG condition
                4. Get the divided data and label based on the split feature and value, and then recursively call GrowTree() to create left and right subtree.
                5. Return the node.  
            Hint: 
                a. You could use the Node class to create a node.
                b. You should carefully deal with the leaf node case. The prediction of the leaf node should be the majority of the labels in this node.

            Args:                                   
                data: Data to be used to train the Decission Tree                           
                label: target variable column name
                counter: counter to keep track of the depth of the tree
            Returns:
                node: New node of the Decission Tree
        '''
        node = Node()
        node.depth = counter

        # Check for depth conditions
        if self.max_depth == None:
            depth_cond = True
        else:
            depth_cond = True if counter < self.max_depth else False

        # Check for sample conditions
        if self.min_samples_split == None:
            sample_cond = True
        else:
            sample_cond = True if data.shape[0] > self.min_samples_split else False

        
        if depth_cond & sample_cond:

            split_feature, split_value, ig = self.BestSplit(data, label)
            node.ig = ig

            # Check for ig condition. If ig condition is fulfilled, make split 
            if ig is not None and ig >= self.min_information_gain:

                node.split_feature = split_feature
                node.split_value = split_value
                counter += 1

                #TODO Get the divided data and label based on the split feature and value, 
                # and then recursively call GrowTree() to create left and right subtree.
                pass

            else:
                # TODO: If it doesn't match IG condition, it is a leaf node
                pass
        else:
            #TODO If it doesn't match depth or sample condition. It is a leaf node
            pass

        return node
    
    def BestSplit(self, data: pd.DataFrame, label: pd.Series):
        '''
            Given a data, select the best split by maximizing the information gain (maximizing the purity)
            Args:
                data: dataframe where to find the best split.
                label: label of the data.
            Returns:
                split_feature: feature to split the data. 
                split_value: value to split the data.
                split_ig: information gain of the split.
        '''
        # TODO: Implement the BestSplit function
        split_feature, split_value, split_ig = None, None, None
        return split_feature, split_value, split_ig

    def predict(self, data: pd.DataFrame) -> List[int]:
        '''
            Given a dataset, make a prediction.
            Args:
                data: data to make a prediction.
            Returns:
                predictions: List, predictions of the data.
        '''
        predictions = []
        # TODO: Implement the predict function
        return predictions
    
    def print_tree(self):
        '''
            Prints the tree.
        '''
        self.print_tree_rec(self.root)

    def print_tree_rec(self, node):
        '''
            Prints the tree recursively.
        '''
        if node is None:
            return 
        else:
            if node.is_leaf:
                print("{}Level{} | Leaf: {}".format(' '* node.depth, node.depth, node.prediction))
                return
            else:
                print("{}Level{} | {} < {} (ig={:0.4f})".format(' '* node.depth, node.depth, node.split_feature, node.split_value, node.ig))
                self.print_tree_rec(node.left)
                self.print_tree_rec(node.right)




def run_train_test(training_data: pd.DataFrame, training_labels: pd.Series, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Args:
        training_data: pd.DataFrame
        training_label: pd.Series
        testing_data: pd.DataFrame
    Returns:
        testing_prediction: List[int]
    """

    #TODO implement the decision tree and return the prediction

    return [1]*len(testing_data)

######################## evaluate the accuracy #################################

def cal_accuracy(y_pred, y_real):
    '''
    Given a prediction and a real value, it calculates the accuracy.
    y_pred: prediction
    y_real: real value
    '''
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    print(sum(y_pred == y_real))
    if len(y_pred) == len(y_real):
        return sum(y_pred == y_real)/len(y_pred)
    else:
        print('y_pred and y_real must have the same length.')

################################################################################

if __name__ == "__main__":
    training = pd.read_csv('data/train.csv')
    dev = pd.read_csv('data/dev.csv')

    print(training)
    training_labels = training['LABEL']
    training_data = training.drop('LABEL', axis=1)
    dev_data = dev.drop('LABEL', axis=1)

    prediction = run_train_test(training_data, training_labels, dev_data)
    accu = cal_accuracy(prediction, dev['LABEL'].to_numpy())
    print(accu)
