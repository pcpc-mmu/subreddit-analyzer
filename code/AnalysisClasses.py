import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
import time
from textacy import preprocessing

"""
from SetUpper import Settings
from MongoLoader import MongoLoader

cfg = Settings()
from pymongo import MongoClient
client = MongoClient(cfg.MONGO_URI)  #  "mongodb://localhost:27017/")
database = client[cfg.DATABASE_NAME]  #  "csvlocal"]
collection = database[cfg.set_all]  # "set_all"
"""

def epochise_datestring(our_date):
    #print(time.strptime(our_date, "%Y-%m-%d %H:%M:%S" ) )
    return time.strptime(our_date, "%Y-%m-%d %H:%M:%S" )


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


    def TextCleaner(text):
        #@TODO chopping of don't/'s/I'm - decontraction
        
        text = preprocessing.normalize_hyphenated_words(text) # fix hyphenated words (possibly not a problem here)
        text = preprocessing.replace.replace_emojis(text,'') # replace/remove emojis
        text = preprocessing.replace.replace_urls(text,'') # replace urls
        text = preprocessing.replace.replace_user_handles(text, '') # remove @user handles
        text = text.replace("\n", " ") # replace newlines with spaces
        text = re.sub(r'^RT\s+' , '', text) # replace/remove RT text 
        text = preprocessing.replace.replace_currency_symbols(text, '') # remove currency symbols
        text = preprocessing.replace.replace_emails(text, '') # replace/remove email strings
        text = preprocessing.replace.replace_numbers(text, '') # replace/remove numbers
        text = preprocessing.remove.remove_punctuation(text) # remove punctuation 
        text = preprocessing.remove.remove_accents(text) # remove accents from text 
        text = preprocessing.normalize_whitespace(text) # tidy up remaining whitespace
        text = text.strip() # remove leading and trailing whitespace
        text = text.lower()
        return text




    def DateFilterADF(dataframe, min_date="", max_date=""):
        our_min_date = epochise_datestring("1970-01-01 00:00:01")
        our_max_date = epochise_datestring("2929-12-31 23:59:59")
        if min_date != "":
            if(isinstance(min_date,str)):
                our_min_date = epochise_datestring(min_date)
            else:
                our_min_date = min_date
            print("min date is {}".format(our_min_date) )
        if max_date != "":
            if(isinstance(max_date,str)):
                our_max_date = epochise_datestring(max_date)
            else:
                our_max_data = max_date
            print("max date is {}".format(our_max_date) )

        rows_to_kill = []
        
        for index,row in dataframe.iterrows():
            row_date = epochise_datestring(row["datetime"])
            if(row_date>=our_min_date)and(row_date<=our_max_date):
                pass
            else:
                rows_to_kill.append(index)
        newdf = dataframe.drop(dataframe.index[rows_to_kill])
        return newdf #dataframe

        
    def DataframeExtractor(dataframe, column, columnvalue, textonly='yes'):
        # duplicated outside class, check which one works? @TODO
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
            text = ' '.join(selection['body'].astype(str).values.flatten().tolist())
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
            
            
    """
    WordCounter
    dataframe - a dataframe! of reddit comments with columns X,Y,X
    column - the column to match column_value's row against
    column_value - a list of names to match
    """          
    def WordCounter(dataframe, column, column_value):
        totalwords = []
        #ac = AnalysisClass()
        for n in column_value:
            selection = DataframeExtractor(dataframe, column, [n])
            text = ' '.join(selection['body'].astype(str).values.flatten().tolist())
            text = AnalysisClasses.TextCleaner(text)
            words = text.split()
            totalwords.append(words)
        total = [item for items in totalwords for item in items]
        counts = Counter(total)
        #print("counts", counts)
        #print("totalwords", totalwords)
        
        totalwordscount = len(totalwords[0]) #somehow double brackets in list
        #print("totalwordcounts",totalwordscount)
        return (counts,totalwordscount)

        
    

        

        
    #def EmbeddingComparer(modellist, modellistnames, targetword, targetnumber, headandtailtype=None, headandtailnrs=10):
    def EmbeddingComparer(modellist, modellistnames):
        out = []
        for model, name in zip(modellist, modellistnames):
            rank = model.wv.most_similar(positive=[str(targetword)], negative=[], topn=int(targetnumber))
            df = pd.DataFrame(rank, columns=['word', 'score'])
            df['rank'] = range(1, len(df) + 1)
            df['model'] = name
            out.append(df)
        df_1 = out[0]
        df_2 = out[1]

        out_1 = []
        for row1, row2 in zip(df_1.itertuples(), df_2.itertuples()):
            # df = pd.DataFrame((getattr(row1, 'model')), columns=['name_1'], index=range(1))
            df = pd.DataFrame((getattr(row1, 'model')), columns=['name_1'], index=range(1))
            df['word_1'] = (getattr(row1, 'word'))
            df['word_1'] = (getattr(row1, 'word'))
            df['score_1'] = (getattr(row1, 'score'))
            df['name_2'] = (getattr(row2, 'model'))
            df['word_2'] = (getattr(row2, 'word'))
            df['score_2'] = (getattr(row2, 'score'))
            out_1.append(df)
        out_1 = pd.concat(out_1, axis=0)

        part1 = out_1.iloc[:, 0:3]
        part2 = out_1.iloc[:, 3:6]
        out_2 = pd.concat([part1, part2], ignore_index=True)

        w1 = out_2.loc[out_2['word_1'].isin(out_2['word_2'])]
        w1 = w1.loc[w1['name_1'] == modellistnames[0]]
        w2 = out_2.loc[out_2['word_2'].isin(out_2['word_1'])]
        w2 = w2.loc[w2['name_2'] == modellistnames[1]]

        out_2['score_1_final'] = out_2['word_1'].apply(
            lambda x: (((w1.loc[w1['word_1'] == x, 'score_1'].iloc[0])) - (
            w2.loc[w2['word_2'] == x, 'score_2'].iloc[0]))
            if (str(x)) in w1['word_1'].tolist()
            # else out_2.loc[out_2['word_1'] == x, ['name_1'].values
            else out_2.loc[out_2['word_1'] == x, ['name_1', 'score_1'][1]].to_string(index=False))

        out_2['score_2_final'] = out_2['word_2'].apply(
            lambda x: (((w2.loc[w2['word_2'] == x, 'score_2'].iloc[0])) - (
            w1.loc[w1['word_1'] == x, 'score_1'].iloc[0]))
            if (str(x)) in w2['word_2'].tolist()
            else out_2.loc[out_2['word_2'] == x, ['name_2', 'score_2'][1]].to_string(index=False))
        # print(out_2)

        pd.options.mode.chained_assignment = None
        total1 = out_2[:targetnumber]  # OPPOSITE FOR TWO
        total1['score_1_final'] = total1['score_1_final'].astype(float)
        total1['index_original'] = total1.index
        total1.drop(['name_2', 'word_2', 'score_2', 'score_2_final'], axis=1, inplace=True)  # OPPOSITE FOR TWO
        total1 = total1.sort_values('score_1_final', ascending=False)
        head1 = total1.head(headandtailnrs)
        tail1 = total1.tail(headandtailnrs)
        headandtail1 = pd.concat([total1[:(int(headandtailnrs / 2))], total1[-(int(headandtailnrs / 2)):]])

        total2 = out_2[targetnumber:]  # OPPOSITE FOR TWO
        total2['score_2_final'] = total2['score_2_final'].astype(float)
        total2['index_original'] = total2.index
        total2.drop(['name_1', 'word_1', 'score_1', 'score_1_final'], axis=1, inplace=True)
        total2 = total2.sort_values('score_2_final', ascending=False)
        head2 = total2.head(headandtailnrs)
        tail2 = total2.tail(headandtailnrs)
        headandtail2 = pd.concat([total2[:(int(headandtailnrs / 2))], total2[-(int(headandtailnrs / 2)):]])

        if headandtailtype == 'head1':
            return head1
        elif headandtailtype == 'tail1':
            return tail1
        elif headandtailtype == 'headandtail1':
            return headandtail1
        elif headandtailtype == 'head2':
            return head2
        elif headandtailtype == 'tail2':
            return tail2
        elif headandtailtype == 'headandtail2':
            return headandtail2
        else:
            return head1, head2


 

