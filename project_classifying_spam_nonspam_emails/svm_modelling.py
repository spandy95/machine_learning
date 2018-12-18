#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Project Name: Spam-NoSpam-Email-Classification
#File Name: svm_modelling.py
#Description: This file prepares the Modelling technique to predict the spam emails
#Inputs: data: Dataframe
#Outputs:
#----------------------------------------------------
#--------------------------------------------------------------------------------------

#...........Importing Libraries...........#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import  model_selection, naive_bayes, metrics, svm
from feature_engineering import get_features_sparse_matrix
from read_data import read_data
from IPython.display import Image
#--------------------------------------------------------------------------------------
# Goal: is to predict if a new sms is spam or non-spam.
#
# we normally don't check the spam messages.
# The two possible situations are:
#     New spam sms in my inbox. (False negative).
#     OUTCOME: I delete it.
#
#     New non-spam sms in my spam folder (False positive).
#     OUTCOME: I probably don't read it.
#--------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------
#-------Split the data in to Training and Testing---------#
#Inputs: X: Sparse Matrix
#--------------------------------------------------------------------------------------
def split_data(X, data):
    print ("\n\n#-------Split the data in to Training and Testing---------#")
    data["v1"] = data["v1"].map({'spam': 1, 'ham': 0})
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
    print ("X_train.shape: {}\nX_test.shape: {}\ny_train.shape: {}\ny_test.shape : {}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    # print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return (X_train, X_test, y_train, y_test)

#--------------------------------------------------------------------------------------
#-------Create amd Train the Model---------#
#Inputs: Training Data and Labels
#--------------------------------------------------------------------------------------
def train_model(X_train, y_train):
    print ("\n\n#-------Create amd Train the Model---------#")
    print("\n#-------Training Inprogress....---------#")
    model = svm.SVC()
    model.fit(X_train, y_train)
    print ("\n\n#-------Training is Completed---------#")
    return model

#--------------------------------------------------------------------------------------
#-------Evaluate Model---------#
#Inputs: model object
#--------------------------------------------------------------------------------------
def evaluate_model(model, X_test,  y_test):
    print ("\n\n#-------Evaluate Model---------#")
    acc = model.score(X_test, y_test)
    print("\nModel Accuracy:", acc)
    return acc




#--------------------------------------------------------------------------------------
#-------Main Program---------#
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = read_data(inputfile)  # fetch the dataset in to dataframe
    X, f = get_features_sparse_matrix(data)

    X_train, X_test, y_train, y_test = split_data(X, data)  # Split the data in to training and testing

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

