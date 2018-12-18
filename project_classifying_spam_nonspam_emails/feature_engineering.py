#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Project Name: Spam-NoSpam-Email-Classification
#File Name: feature_engineering.py
#Description: This file prepares important features
#Inputs: data: Dataframe
#Outputs:
#----------------------------------------------------
#--------------------------------------------------------------------------------------

#...........Importing Libraries...........#
from sklearn import feature_extraction
import read_data as rd




# Feature engineering:
# We remove the stop words in order to improve the analytics ie, from column v2 remove the stop words

# CountVectorizer :
# this function helps to split the data in to words also helps to remove the stop words
# pass the column data as the input the function itself splits the data in to words and
# removes the stop words

#--------------------------------------------------------------------------------------
#-------Important Features are extracted by removing the Stop Words---------#
#--------------------------------------------------------------------------------------
 ## -  there will be 5572 lines in the file
def get_features_sparse_matrix(data):
    print ("\n\n#-------Important Features are extracted by removing the Stop Words---------#")
    f = feature_extraction.text.CountVectorizer(stop_words='english')
    X = f.fit_transform(data["v2"])

    ## -  there will be 5572 lines in the file
    print("Shape of the Sparse Matrix:", X.shape)

    print("\n\n#-------printing first 10 values of Sparse Matrix---------#")
    for i in range(10):
        print(X[i])
    return (X, f)

#--------------------------------------------------------------------------------------
#-------Newly Created Features---------#
#--------------------------------------------------------------------------------------
def newly_created_features(X, f):
    print("\n\n#-------Newly Created Features---------#\n#-------Printing first 20 Features---------#")
    feature_names = f.get_feature_names()
    print ("There is a total of {} features:".format(len(feature_names)))
    for i in range(20):
        print(feature_names[i])

#--------------------------------------------------------------------------------------
#-------Main Program---------#
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = rd.read_data('spam.csv')  # fetch the dataset in to dataframe
    X, f = get_features_sparse_matrix(data)

    #We have created more than 8400 new features.
    newly_created_features(X, f)


