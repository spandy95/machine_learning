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
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn import  model_selection, naive_bayes, metrics, svm
from feature_engineering import get_features_sparse_matrix
from read_data import read_data
from svm_modelling import split_data
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
#-------Create amd Train the Model---------#
#Inputs: Training Data and Labels
#--------------------------------------------------------------------------------------
def train_model(X_train, y_train):
    print ("\n\n#-------Create amd Train the Model---------#")
    print("\n#-------Training Inprogress....---------#")
    model = MultinomialNB()
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



