# Starter code for CS 165B MP1 Spring 2023
import numpy as np
def find_t(centroid_1,centroid_2):
    return (1/2)*np.dot((centroid_1 + centroid_2),(centroid_1 - centroid_2))

def find_centroid(dim,data):
    output = np.zeros(dim) 
    for i in range(len(data)):
        output += data[i]
    return output/len(data)
        
def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the dimpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """


    # TODO: IMPLEMENT
    pass
    # The dimensions of the input data sets
    dim = training_input[0][0]
    dim_a = training_input[0][1]
    dim_b = training_input[0][2]
    dim_c = training_input[0][3]
    
    # The data sets sliced from the input array
    data_a = training_input[1:dim_a+1]
    data_b = training_input[dim_a + 1: dim_a + dim_b+1]
    data_c = training_input[dim_a + dim_b + 1:]

    # Getting the centroid vectors from each data class
    centroid_a = np.array(find_centroid(dim,data_a))
    centroid_b = np.array(find_centroid(dim,data_b))
    centroid_c = np.array(find_centroid(dim,data_c))

    # Calculating w for the separation of each of the data classes
    w_ab = centroid_a - centroid_b
    w_bc = centroid_b - centroid_c
    w_ac = centroid_a - centroid_c
    t_ab = find_t(centroid_a,centroid_b)
    t_bc = find_t(centroid_b,centroid_c)
    t_ac = find_t(centroid_a,centroid_c)

    # Getting the dimension of the testing data set
    dim_a_testing = testing_input[0][1]
    dim_b_testing = testing_input[0][2]
    dim_c_testing = testing_input[0][3]

    # Creating entries of the table. AB = Actual A, Predicted B. BC = Actual B, Predicted C,...
    AA, AB, AC, BA, BB, BC, CA, CB, CC = 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Iterating over the testing input and classifying based on obtained data from training
    for i in range(1,len(testing_input[1:])):
        # decides that its class a or c
        if(np.dot(testing_input[i],w_ab) >= t_ab):
            # decides that its class a
            if(np.dot(testing_input[i],w_ac) >= t_ac):
                if(i <= dim_a_testing):
                    AA += 1
                elif(i <= dim_a_testing+dim_b_testing):
                    BA += 1
                else:
                    CA += 1
            # decides that its class c
            else:
                if(i <= dim_a_testing):
                    AC += 1
                elif(i <= dim_a_testing+dim_b_testing):
                    BC += 1
                else:
                    CC += 1
        # decides that its class b or c
        else:
            # decides that its class b
            if(np.dot(testing_input[i],w_bc) >= t_bc):
                if(i <= dim_a_testing):
                    AB += 1
                elif(i <= dim_a_testing+dim_b_testing):
                    BB += 1
                else:
                    CB += 1
            # decides that its class c
            else:
                if(i <= dim_a_testing):
                    AC += 1
                elif(i <= dim_a_testing+dim_b_testing):
                    BC += 1
                else:
                    CC += 1

    total = AA+AB+AC+BA+BB+BC+CA+CB+CC
    # true positive rate = true positive / (true positive + false negative)
    TPR_A = AA / (AA+AB+AC)
    TPR_B = BB / (BB+BA+BC)
    TPR_C = CC / (CA+CB+CC)
    TPR_AVG = (TPR_A+TPR_B+TPR_C)/3

    # false positive rate = false positive / (false positive + true negative)
    FPR_A = (BA+CA) / (BA+CA+BB+CB+BC+CC)
    FPR_B = (AB+CB) / (AB+CB+AA+CA+AC+CC)
    FPR_C = (AC+BC) / (AC+BC+AA+BA+AB+BB)
    FPR_AVG = (FPR_A + FPR_B + FPR_C) / 3

    # error rate = (false positive + false negative) / total
    ER_A = (BA+CA+AB+AC) / total
    ER_B = (AB+CB+BA+BC) / total
    ER_C = (AC+BC+CA+CB) / total
    ER_AVG = (ER_A + ER_B + ER_C) / 3

    # accuracy = 1 - error rate
    ACC_AVG = 1 - ER_AVG
    
    # precision = true positives / (true positives + false positives)
    PR_A = AA / (AA+BA+CA)
    PR_B = BB / (BB+AB+CB)
    PR_C = CC / (CC+AC+BC)
    PR_AVG = (PR_A+PR_B+PR_C)/3

    return {
        "tpr": TPR_AVG,
        "fpr": FPR_AVG, 
        "error_rate": ER_AVG,
        "accuracy": ACC_AVG,
        "precision": PR_AVG
    }
