#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Project Name: Spam-NoSpam-Email-Classification
#File Name: exploratory_analysis.py
#Description: This file explain the data with different charts
#Inputs: data: Dataframe
#Outputs: Charts
#----------------------------------------------------
#--------------------------------------------------------------------------------------

#...........Importing Libraries...........#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import Image
import read_data as rd
#--------------------------------------------------------------------------------------
#-------Distribution spam/non-spam plots---------#
#--------------------------------------------------------------------------------------
def freq_plot(data):
    print ("\n\n#----------- Distribution spam/non-spam plots-----------#")
    count_Class=pd.value_counts(data["v1"], sort= True)
    print (count_Class)
    return(count_Class)


#--------------------------------------------------------------------------------------
#-------Plot pie Chart for Ham and Spam counts---------#
#--------------------------------------------------------------------------------------
def pie_chart(count_Class):
    print ("\n\n#-------Plot pie Chart for Ham and Spam counts---------#")
    count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
   # plt.savefig('charts/pie_chart.png')
    plt.show()

# Text Analytics
# We want to find the frequencies of words in the spam and non-spam messages. The words of the messages will be model features.
# We use the function Counter.
# Pick V2 column and split in to words for "ham" or "spam records
#--------------------------------------------------------------------------------------
#-------Text Analytics-Most Common of Words in ham text---------#
# Inputs: data: Dataframe, typeofmsg : "ham" or "spam"
#--------------------------------------------------------------------------------------
def common_words(data, typeofmsg):
    allwords = ' '.join(data[data['v1'] == typeofmsg]['v2']).split()
    Counter(allwords)
    Counter(allwords).most_common(20)
    count1 = Counter(" ".join(data[data['v1']==typeofmsg]["v2"]).split()).most_common(20)
    #### Convert the the most common words as the dataframe
    df1 = pd.DataFrame.from_dict(count1)
    if typeofmsg == "ham":
        msg = "words in non-spam"
    else:
        msg = "words in spam"
    df1 = df1.rename(columns={0: msg, 1: "count"})
    print("\n\n#-------Text Analytics-Most Common of Words in {} text---------#".format(msg))
    print(df1)
    return df1


#--------------------------------------------------------------------------------------
#-------Pie charts---------#
# Inputs: data: Dataframe, typeofmsg : "ham" or "spam"
#--------------------------------------------------------------------------------------
def bar_plots(df1, typeofmsg):
    if typeofmsg == 'ham':
        field = "words in non-spam"
    else:
        field = "words in spam"
    df1.plot.bar()
    y_pos = np.arange(len(df1[field]))
    plt.xticks(y_pos, df1[field])
    plt.title('More frequent words in {} messages'.format(field))
    plt.xlabel('words')
    plt.ylabel('number')
    fig_name = 'bar_chart_'+typeofmsg+'.png'
    fig_name = 'charts/'+fig_name
    plt.savefig(fig_name)
    plt.show()

#--------------------------------------------------------------------------------------
#-------Main Program---------#
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    data = rd.read_data("spam.csv")
    print(data.head(10))

    count_Class = freq_plot(data) #  frequency plot
    pie_chart(freq_plot(data))  #pie chart

    df_ham = common_words(data, 'ham') # print most common words in ham or spam text
    df_spam = common_words(data, 'spam')  # print most common words in ham or spam text

    bar_plots(df_ham, 'ham') # print bar_plots for most commonwords in ham messages

    bar_plots(df_spam, 'spam')  # print bar_plots for most commonwords in ham messages

    print ("\n\n---------------------END of Exploratory Analysis---------------------")



