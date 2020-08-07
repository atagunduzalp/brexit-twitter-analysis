import GetOldTweets3 as got
from nltk import tokenize
import nltk
# nltk.download('stopwords')
import os
import os.path
import string
from snowballstemmer import stemmer
from nltk.stem import PorterStemmer
import csv

from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split

import model
import getTweets

AFTER_NORMALIZATION_CSV = "tweets-after-normalization.csv"

f = open(AFTER_NORMALIZATION_CSV, 'w')
writer = csv.writer(f)
writer.writerow(['tweet', 'sentiment'])

def read_csv():
    data = pd.read_csv("original-tweets.csv")
    return data

def write_to_CSV(tweet, sentiment):
    text_class_list = []
    text_class_list.append(tweet)
    text_class_list.append(sentiment)
    writer.writerow(text_class_list)

def normalize_tweets(data):
    normalized_tweets = []
    word_dict = {}
    tweet_list = []
    for index in data.index:
        if data['Sentiment'][index] != 2:
            tweet = data['Tweet'][index]
            # print("original tweet: " + str(tweet))
            tweet, before_after_dict = split_into_sentences(tweet)
            write_to_CSV(tweet, data['Sentiment'][index])
            tweet_list.append(tweet)
            create_dictionary(tweet, word_dict)
    f.close()
    sorted_dict= {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
        # print(str(sorted_dict))
    return tweet_list, sorted_dict

#***************************
# 1 Metni cümlelere ayırın #
#***************************
def split_into_sentences(text):
    # split paragraph into sentences with the help of nltk library.
    # https://kite.com/python/answers/how-to-split-text-into-sentences-in-python
    sentences_list_nltk = tokenize.sent_tokenize(text)
    return lower_case_context(sentences_list_nltk)

#**********************************
# 2 Metnin küçük harfe çevrilmesi #
#**********************************
def lower_case_context(sentences):
    lower_cased_sentences = []
    # print(sentences)

    for sentence in sentences:
        lower_cased_sentences.append(str(sentence).lower())
    return cut_apostrophe(lower_cased_sentences)

#*****************************************************
# 5 Kesme işaretinden sonraki harflerin temizlenmesi #
#*****************************************************
def cut_apostrophe(sentences):
    all_sentences = []
    for sentence in sentences:
        words = sentence.split()
        word_list = []
        for word in words:
            if "’" in word:
                new_word = ""
                for char in word:
                    if char != "’":
                        new_word += char
                    else:
                        break   
                word_list.append(new_word)
            else:
                word_list.append(word)
            text = ' '.join(word_list)
        all_sentences.append(text)
    return remove_punctuations(all_sentences)

#*****************************************
# 3 Noktalama işaretlerinin temizlenmesi #
#*****************************************
def remove_punctuations(sentences):
    punctuation = ""
    for char in string.punctuation:
        punctuation += char
    # https://machinelearningmastery.com/clean-text-machine-learning-python/
    punctuation += '”'
    punctuation += '“'
    punctuation += "\u2026"
    punctuation += '‘'
    table = str.maketrans('', '', punctuation)
    strippted_lowercased_text = []
    for sentence in sentences:
        words = sentence.split()
        word_list = []
        for word in words:
            strippted_word = word.translate(table)
            word_list.append(strippted_word)
        text = ' '.join(word_list)
        strippted_lowercased_text.append(text)
    return delete_numeric(strippted_lowercased_text)

#***************************
# 4 Sayıların temizlenmesi #
#***************************
def delete_numeric(sentences):
    non_numerical_words = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 1:
            word_list = []
            for word in words:
                word_or_not = str(word).isnumeric()
                if word_or_not or word.startswith('http'):
                    continue
                else:
                    word_list.append(word)
                text = ' '.join(word_list)
            non_numerical_words.append(text)
    return delete_stopwords(non_numerical_words)

#*************************************
# 6 Durma kelimelerinin temizlenmesi #
#*************************************
def delete_stopwords(sentences):
    # https://pythonspot.com/nltk-stop-words/
    stop_word_list = nltk.corpus.stopwords.words('english')
    stop_word_list.append("amp")
    all_sentences = []
    stop_words = []
    for sentence in sentences:
        words = sentence.split()
        word_list = []
        if len(words) > 0:
            for word in words:
                if word in stop_word_list:
                    stop_words.append(word)
                    continue
                else:
                    word_list.append(word)
                text = ' '.join(word_list)
            if len(word_list) > 0:
                all_sentences.append(text)
    return seperate_with_dash(all_sentences)

#*******************************************************************************
# 7 çizgiyle ayrılan kelimelerin ayrı ayrı kelimeler olarak değerlendirilmesi. #
#*******************************************************************************
def seperate_with_dash(sentences):
    all_sentences = []
    for sentence in sentences:
        words = sentence.split()
        word_list = []
        for  word in words:
            if "-" in word:
                splitted_words = word.split('-')
                new_words = " ".join(splitted_words)
                word_list.append(new_words)
            else:
                word_list.append(word)
            text = ' '.join(word_list)
        all_sentences.append(text)
    return parse_with_porter(all_sentences)

#*************************************
# 8 Kelime uzunluğuna göre gövdeleme #
#*************************************        
def parse_words_manuel(sentences):
    all_sentences = []
    for sentence in sentences:
        words = sentence.split()
        word_list = []
        for word in words:
            if len(word) > 5:
                word_list.append(word[0:5])
            else:
                word_list.append(word)
            text = ' '.join(word_list)
        all_sentences.append(text)
    return all_sentences

#**********************************
# 9 snowballstemmer ile gövdeleme #
#**********************************
# https://medium.com/@aanilkayy/pythonda-snowball-stemmer-kullan%C4%B1lmas%C4%B1-e91ed9be8e9e
# def parse_with_snowball(sentences):
#     before_after_dict = {}
#     turkish_stemmer = stemmer("English")
#     all_sentences = []
#     for sentence in sentences:
#         words = sentence.split()
#         word_list = []
#         for word in words:
#             stemmed_word = turkish_stemmer.stemWord(word)
#             before_after_dict[word] = stemmed_word
#             word_list.append(stemmed_word)
#             text = ' '.join(word_list)
#         all_sentences.append(text)
#     return all_sentences, before_after_dict

def parse_with_porter(sentences):
    before_after_dict = {}
    ps = PorterStemmer()
    all_sentences = []
    for sentence in sentences:
        words = sentence.split()
        word_list = []
        for word in words:
            stemmed_word = ps.stem(word)
            before_after_dict[word] = stemmed_word
            word_list.append(stemmed_word)
            text = ' '.join(word_list)
        all_sentences.append(text)
    tweet = ""
    # for sentence in all_sentences:
    tweet = ' '.join(all_sentences)
    return tweet, before_after_dict

#******************************
# 10 kelime sözlüğü oluşturma #
#******************************
def create_dictionary(sentence, word_dict):
    words = sentence.split()
    for word in words:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
    return word_dict


if __name__ == "__main__":
    '''In order get train tweets dataset'''
    # politicians = ['conservatives', "David_Cameron", "BorisJohnson", "theresa_may", "UKIP", "Nigel_Farage", 
    #               "_HenryBolton", "DouglasCarswell", "blaiklockBP", "brexitparty_uk", "june_mummery", 
    #               "benhabib6", "BrexitAlex", "UKLabour", "jeremycorbyn", "tom_watson", "stellacreasy", "IainMcNicol", "JennieGenSec",
    #             "LibDems", "timfarron", "vincecable", "joswinson", "TheGreenParty", "natalieben", "sianberry", 
    #             "jon_bartley", "theSNP", "StewartHosieSNP", "NicolaSturgeon", "AngusRobertson"]
    # getTweets.getTweets(politicians, 1)

    data = read_csv()
    normalize_tweets(data)
    model.read_file_from_csv(AFTER_NORMALIZATION_CSV)
