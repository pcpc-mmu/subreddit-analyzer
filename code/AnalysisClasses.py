import pandas as pd
import re
import matplotlib.pyplot as plt

"""
from SetUpper import Settings
from MongoLoader import MongoLoader

cfg = Settings()
from pymongo import MongoClient
client = MongoClient(cfg.MONGO_URI)  #  "mongodb://localhost:27017/")
database = client[cfg.DATABASE_NAME]  #  "csvlocal"]
collection = database[cfg.set_all]  # "set_all"
"""



def DataframeExtractor(dataframe, column, columnvalue, textonly='no'):
    #selection = dataframe[dataframe[column] == columnvalue]
    selection = dataframe[dataframe[column].isin(columnvalue)]
    #print(selection)
    if textonly == 'yes':
        textselection = selection['body']
        return textselection
    else:
        return selection




class AnalysisClasses:

    def __init__(self):
        self.accounts = []
        
        
    def DataframeExtractor(dataframe, column, columnvalue, textonly='yes'):
        selection = dataframe[dataframe[column] == columnvalue]
        if textonly == 'yes':
            textselection = selection['body']
            return textselection
        else:
            return out
        

    def VaderAnalyser(text):
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(text)
        #print(scores)
        return(scores)
       
    
    
    def SentimentGrapher(dataframe, column, column_value, plotkind='bar', plotname=None):
        out = []
    
        neg_all = []
        neu_all = []
        pos_all = []
        for n in column_value:
            selection = DataframeExtractor(dataframe, column, [n])
            text = ''.join(selection['body'].astype(str).values.flatten().tolist())
            sentiment = AnalysisClasses.VaderAnalyser(text)
            neg_all.append(float(sentiment["neg"]))
            neu_all.append(float(sentiment["neu"]))
            pos_all.append(float(sentiment["pos"]))
            
            neg_sum = sum(neg_all)
            neu_sum = sum(neu_all)
            pos_sum = sum(pos_all)
            if len(neg_all) == 0:
                neg_avg = 0
            else:
                neg_avg = neg_sum / len(neg_all)
            if len(neu_all) == 0:
                neu_avg = 0
            else:
                neu_avg = neu_sum / len(neu_all)
            if len(pos_all) == 0:
                pos_avg = 0
            else:
                pos_avg = pos_sum / len(pos_all)
            #print(n, neg_avg)

            df = pd.DataFrame(n, columns=[column], index=range(1))
            df['neg_avg'] = neg_avg
            df['neu_avg'] = neu_avg
            df['pos_avg'] = pos_avg
            out.append(df)
        outout = pd.concat(out, axis=0)
        #subdf = outout[1:3]
        
        if plotname:
            plot = outout.plot(kind=plotkind, x=column, y=['neg_avg', 'pos_avg'])
            plot.set_title('Sentiment per ' + column)
            plt.show()
            plt.savefig(plotname + '.png')
            return outout
        else:
            return outout
            
        



        
        
        
        
            
#     def SentimentGrapher(dataframe, column, column_value, plotkind='bar', plotname=None):
#         out = []
        
#         neg_all = []
#         neu_all = []
#         pos_all = []
#         selection = DataframeExtractor(dataframe, column, column_value)
#         for index, row in selection.iterrows():
#             text = row['body']
#             sentiment = AnalysisClasses.VaderAnalyser(text)
#             #print(sentiment)
#             neg_all.append(float(sentiment["neg"]))
#             neu_all.append(float(sentiment["neu"]))
#             pos_all.append(float(sentiment["pos"]))

#             neg_sum = sum(neg_all)
#             neu_sum = sum(neu_all)
#             pos_sum = sum(pos_all)
#             if len(neg_all) == 0:
#                 neg_avg = 0
#             else:
#                 neg_avg = neg_sum / len(neg_all)
#             if len(neu_all) == 0:
#                 neu_avg = 0
#             else:
#                 neu_avg = neu_sum / len(neu_all)
#             if len(pos_all) == 0:
#                 pos_avg = 0
#             else:
#                 pos_avg = pos_sum / len(pos_all)
#             #print(neg_avg)

#             df = pd.DataFrame(row, columns=[column], index=range(1))
#             df['neg_avg'] = neg_avg
#             df['neu_avg'] = neu_avg
#             df['pos_avg'] = pos_avg
#             #print(df)
#             out.append(df)
#         outout = pd.concat(out, axis=0)
#         return outout
#         #subdf = outout[1:3]
        
#         if plotname:
#             plot = outout.plot(kind=plotkind, x=column, y=['neg_avg', 'pos_avg'])
#             plot.set_title('Sentiment per ' + column)
#             plt.show()
#             plt.savefig(plotname + '.png')
#             return outout
#         else:
#             return outout
        
        


 

