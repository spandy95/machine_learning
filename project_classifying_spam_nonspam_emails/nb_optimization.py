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
from svm_modelling import split_data,evaluate_model

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
#----------------------------finding best index and corresponding c value--------------
#--------------------------------------------------------------------------------------
def optimization(X_train, X_test, y_train, y_test):
    list_alpha = np.arange(1 / 100000, 20, 0.11)
    score_train = np.zeros(len(list_alpha))
    score_test = np.zeros(len(list_alpha))
    recall_test = np.zeros(len(list_alpha))
    precision_test = np.zeros(len(list_alpha))
    count = 0
    for alpha in list_alpha:
        bayes = naive_bayes.MultinomialNB(alpha=alpha)
        bayes.fit(X_train, y_train)
        score_train[count] = bayes.score(X_train, y_train)
        score_test[count] = bayes.score(X_test, y_test)
        recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
        precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
        count = count + 1

    matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
    models = pd.DataFrame(data=matrix, columns=['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])

    best_index = models[models['Test Precision'] == 1]['Test Accuracy'].idxmax()
    print(models.iloc[best_index, :])
    print("the best alpha value to get maximum accuracy is{}".format(list_alpha[best_index]))
    return(list_alpha[best_index])



#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#-------Create amd Train the Model---------#
#Inputs: Training Data and Labels
#--------------------------------------------------------------------------------------
def train_model(X_train, y_train,alpha):
    print ("\n\n#-------Create amd Train the Model---------#")
    print("\n#-------Training Inprogress....---------#")
    model = naive_bayes.MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    print ("\n\n#-------Training is Completed---------#")
    return( model)


#--------------------------------------------------------------------------------------
#-------Main Program---------#
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = read_data('spam.csv')  # fetch the dataset in to dataframe
    X, f = get_features_sparse_matrix(data)

    X_train, X_test, y_train, y_test = split_data(X, data)  # Split the data in to training and testing
    alpha=optimization(X_train, X_test, y_train, y_test)
    model= train_model(X_train, y_train,alpha)

    evaluate_model(model, X_test, y_test)

