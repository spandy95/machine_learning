#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Project Name: Spam-NoSpam-Email-Classification
#File Name: Read_data.py
#Description: This files read the data in the required format and returns the dataframe
#Inputs: None
#Outputs: data- Dataframe
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


# The simplest text encoding (called 'latin-1' or 'iso-8859-1')
# maps the code points 0–255 to the bytes 0x0–0xff, which means that a
# string object that contains code points above U+00FF
# can’t be encoded with this codec.
# encoding : str, default None Encoding to use for UTF when
# reading/writing (ex. 'utf-8').
# `List of Python standard encodings
# https://docs.python.org/3/library/codecs.html#standard-encodings


#...........Importing Libraries...........#
import pandas as pd
#...........Read Dataset...........#
def read_data(inputfile):
    data = pd.read_csv(inputfile, encoding='latin-1')
    # data = pd.read_csv('spam.csv')
    print (data.head(n=10))
    return data
