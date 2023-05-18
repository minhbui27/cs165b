# Starter code for CS 165B MP2
# Decision Truee


import os
import sys
import json
import numpy as np
import pandas as pd
import math

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
    def __str__(self):
        print("is_leaf: " + self.is_leaf)

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
                
                # getting the left and right datas to pass into GrowTree for left and right nodes
                left_data = data[data[split_feature] < split_value].drop(split_feature,axis = 1)
                right_data = data[data[split_feature] >= split_value].drop(split_feature,axis = 1)
                left_label = label[data[split_feature] < split_value]
                right_label = label[data[split_feature] >= split_value]
                node.left = self.GrowTree(left_data,left_label,counter)
                node.right = self.GrowTree(right_data,right_label,counter)

            else:
                # TODO: If it doesn't match IG condition, it is a leaf node
                # Getting the count of 2's and 1's in the label and returning the majority
                node.is_leaf = True
                _, counts = np.unique(label, return_counts=True)
                if(len(counts) == 0):
                    node.prediction = 1
                else:
                    node.prediction = 2 if counts[1] > counts[0] else 1
        else:
            #TODO If it doesn't match depth or sample condition. It is a leaf node
            # Getting the count of 2's and 1's in the label and returning the majority
            node.is_leaf = True
            _, counts = np.unique(label, return_counts=True)
            if(len(counts) == 0):
                node.prediction = 1
            else:
                node.prediction = 2 if counts[1] > counts[0] else 1

        return node
    

    ###################### getting the entropies ######################
        
    def calculate_entropy(self, data: pd.Series):
        _, counts = np.unique(data, return_counts=True)
        if(len(counts) <= 1):
            return 1
        positives = counts[0] 
        negatives = counts[1] 
        p = positives/(positives+negatives)
        return self.calculate_gini_impurity(p)

    def calculate_entropy_impurity(self, p):
        if(p == 0 or p == 1):
            return 0
        return -p*math.log(p,2) - (1-p)*math.log(1-p,2)
    
    def calculate_gini_impurity(self, p):
        if(p == 0 or p == 1):
            return 0
        return 2*p*(1-p)

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
        split_feature, split_value, split_ig = None, None, 0
        # calculating the split value
        init_entropy = self.calculate_entropy(label)
        for feature in data:
            unique_values = data[feature].unique()
            for i in range(1,len(unique_values)):
                value = (unique_values[i] + unique_values[i-1])/2
                left = label[data[feature] < value]
                right = label[data[feature] >= value]

                left_entropy = self.calculate_entropy(left)
                right_entropy = self.calculate_entropy(right)

                ig = init_entropy - (left_entropy*len(left) + right_entropy*len(right))/len(label)

                if(ig > split_ig):
                    split_ig = ig
                    split_feature = feature
                    split_value = value

        return split_feature, split_value, split_ig

    def predict(self, data: pd.DataFrame) -> List[int]:
        '''
            Given a dataset, make a prediction.
            Args:
                data: data to make a prediction.
            Returns:
                predictions: List, predictions of the data.
        '''
        predictions = [0] * len(data)
        # TODO: Implement the predict function
        for idx, row in data.iterrows():
            predictions[idx] = self.predict_recur(row, self.root)
        return predictions
    
    def predict_recur(self, row: pd.Series, node: Node):
        if(node.is_leaf):
            return node.prediction
        else:
            next_node = node.left if row[node.split_feature] < node.split_value else node.right
            return self.predict_recur(row, next_node)

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
    tree = DecisionTree(2,100,1e-10)
    tree.fit(training_data,training_labels)
    tree.print_tree()
    return tree.predict(testing_data)



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

    training_labels = training['LABEL']
    training_data = training.drop('LABEL', axis=1)
    dev_data = dev.drop('LABEL', axis=1)

    ######## Printing all features in the data to test which feature to clean ########
    # for feature in training_data:
    #     print(feature)
    ######## Doing some cleaning to the data set ########
    feature = 'INTUBED'
    q_low = training_data[feature].quantile(0.01)
    q_hi  = training_data[feature].quantile(0.99)

    df_filtered = training_data[(training_data[feature] < q_hi) & (training_data[feature] > q_low)]
    clean_idx = training_data.index[(training_data[feature] < q_hi) & (training_data[feature] > q_low)].tolist()
    clean_labels = training_labels[clean_idx].reset_index().drop('index',axis = 1)
    clean_training_set = training_data.loc[clean_idx].reset_index().drop('index',axis = 1)
    # print(clean_training_set)
    ######## Getting the prediction and calculating accuracy ########
    prediction = run_train_test(clean_training_set, clean_labels, dev_data)
    accu = cal_accuracy(prediction, dev['LABEL'].to_numpy())
    print(accu)
