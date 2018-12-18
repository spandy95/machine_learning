#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Project Name: Spam-NoSpam-Email-Classification
#File Name: main.py
#Description: This project classifies the Spam and Non-Spam Emails
#----------------------------------------------------
#--------------------------------------------------------------------------------------
# Context
# The SMS Spam Collection is a set of SMS tagged messages
# that have been collected for SMS Spam research.
# It contains one set of SMS messages in English of 5,574 messages,
# tagged according being ham (legitimate) or spam.
#
# Content
# The files contain one message per line.
# Each line is composed by two columns:
# v1 contains the label (ham or spam) and v2 contains the raw text.
# A collection of 425 SMS spam messages was manually extracted
# from the Grumbletext Web site.
# This is a UK forum in which cell phone users make public claims
# about SMS spam messages, most of them without reporting the
# very spam message received. The identification of the text of
# spam messages in the claims is a very hard and time-consuming task,
# and it involved carefully scanning hundreds of web pages.
#
# Problem Description
# Can you use this dataset to build a prediction model that will
# accurately classify which texts are spam?
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#-------Summary of the Project Phases---------#
# - Loading the data
# - Describe and Analyse the data




#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#-------Importing Libraries---------#
#--------------------------------------------------------------------------------------
import read_data as rd
import exploratory_analysis as ea
import matplotlib.pyplot as plt
import feature_engineering as fe
import svm_modelling as sm
import nb_modelling as nbm
import svm_optimization as som
import numpy as np
import nb_optimization as nbo
#--------------------------------------------------------------------------------------
#-------Main Program---------#
#--------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n\n---------------------Project to Classify Spam and Non Spam Emails---------------------")
    print("\n\n---------------------Load Data---------------------")
    data = rd.read_data('spam.csv')
    print(data.head(10))
    print("\n\nLoading Data Got Competed...")
    print("\n\n---------------------Exploratory Analysis Started---------------------")
    count_Class = ea.freq_plot(data)  # frequency plot
    ea.pie_chart(count_Class)  # pie chart

    df_ham = ea.common_words(data, 'ham')  # print most common words in ham or spam text
    df_spam = ea.common_words(data, 'spam')  # print most common words in ham or spam text

    ea.bar_plots(df_ham, 'ham')  # print bar_plots for most commonwords in ham messages

    ea.bar_plots(df_spam, 'spam')  # print bar_plots for most commonwords in ham messages

    print("\n\n---------------------END of Exploratory Analysis---------------------")



    print("\n\n---------------------Feature Engineering Started---------------------")
    X, f = fe.get_features_sparse_matrix(data)
    #We have created more than 8400 new features.6:30 AM 12/3/20186:30 AM 12/3/20186:30 AM 12/3/20186:30 AM 12/3/20186:30 AM 12/3/20186:30 AM 12/3/20186:30 AM 12/3/2018
    fe.newly_created_features(X, f)
    print("\n\nFeature Engineering Completed...")

    print("\n\n---------------------Data Preparation Started---------------------")

    X_train, X_test, y_train, y_test = sm.split_data(X, data)  # Split the data in to training and testing
    print("\n\nData Preparation Completed...")
    print("-----------svm modelling started-------------")
    print("\n\n---------------------Training of svm modelling Started---------------------")
    model = sm.train_model(X_train, y_train)

    print("\n\n---------------------Evaluation Started Started---------------------")

    sm.evaluate_model(model, X_test, y_test)
    print("---------------------------end of svm modelling-----------------------")

    print("---------------optimized svm modelling started------------------------")

    print("\n\n---------------------Training of the optimized Model Started---------------------")
    C=som.optimization(X_train, X_test, y_train, y_test)
    model = som.train_model(X_train, y_train, C)

    print("\n\n---------------------Evaluation Started---------------------")
    som.evaluate_model(model, X_test, y_test)

    print("we can observe that the accuracy of svm model increased by adding c value")
    print("--------optimised svm modelling ended--------------------")

    print("------------------------nb modelling started--------------------------")
    print("\n\n---------------------Train the Model Started---------------------")
    model = nbm.train_model(X_train, y_train)

    print("\n\n---------------------Evaluation Started Started---------------------")
    nbm.evaluate_model(model, X_test, y_test)


    print("\n\n*********************End of the nbm******************************")
    print("--------start of nb optimized modelling--------------------------- ")
    alpha=nbo.optimization(X_train, X_test, y_train, y_test)
    print("---------------------optimised training of nb model started---------------------------")
    model=nbo.train_model(X_train, y_train, alpha)
    print("---------------evalution of optimised nbm-----------------------")
    nbo.evaluate_model(model,X_test, y_test)
    print("-------------------- End of nb optimized modelling-------------------")
   # print("we can observe that even after optimizing the naive bayes modelling we r getting same accuracy")
    print("\n\n*********************End of the Project******************************")
