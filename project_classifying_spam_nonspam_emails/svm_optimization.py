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
from collections import Counter
from sklearn import  model_selection, naive_bayes, metrics, svm
from feature_engineering import get_features_sparse_matrix
from read_data import read_data
from svm_modelling import split_data
import svm_modelling as sm
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
#----------------------------finding best index and corresponding c value--------------
#--------------------------------------------------------------------------------------
def optimization(X_train, X_test, y_train, y_test):
    list_C = np.arange(500, 2000, 100)
    score_train = np.zeros(len(list_C))
    score_test = np.zeros(len(list_C))
    recall_test = np.zeros(len(list_C))
    precision_test = np.zeros(len(list_C))
    count = 0

    for C in list_C:
        svc = svm.SVC(C=C)
        svc.fit(X_train, y_train)
        score_train[count] = svc.score(X_train, y_train)
        score_test[count] = svc.score(X_test, y_test)
        recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
        precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
        count = count + 1
    models = pd.DataFrame({'C': list_C,
                           'Train Accuracy': score_train,
                           'Test Accuracy': score_test,
                           'Test Recall': recall_test,
                           'Test Precision': precision_test})
    best_index = models[models['Test Precision'] == 1]['Test Accuracy'].idxmax()
    print(models.iloc[best_index, :])
    print(list_C[best_index])
    C = list_C[best_index]
    return (C)


#--------------------------------------------------------------------------------------
#-------Create amd Train the Model---------#
#Inputs: Training Data and Labels
#--------------------------------------------------------------------------------------
def train_model(X_train, y_train, C):
    print ("\n\n#-------Create amd Train the Model---------#")
    print("\n#-------Training Inprogress....---------#")
    model = svm.SVC(C=C)
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

#--------------------------------------------------------------------------------------
#-------Main Program---------#
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = read_data(inputfile)# fetch the dataset in to dataframe
    X, f = get_features_sparse_matrix(data)

    X_train, X_test, y_train, y_test = split_data(X, data)  # Split the data in to training and testing
    C=optimization(X_train, X_test, y_train, y_test)
    model = train_model(X_train, y_train,C)

    evaluate_model(model, X_test, y_test)
    m_confusion_test = metrics.confusion_matrix(y_test, model.predict(X_test))
    print(pd.DataFrame(data=m_confusion_test, columns=['Predicted 0', 'Predicted 1'],
                 index=['Actual 0', 'Actual 1']))
