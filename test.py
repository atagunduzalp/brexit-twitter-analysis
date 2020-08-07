import GetOldTweets3 as got
import csv
import pickle
import data_cleaner
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

filename = "finalized_model.sav"

f = open('test.csv', 'w')
writer = csv.writer(f)
writer.writerow(['tweet', 'sentiment'])

def writeTweetsIntoCSV(tweet, sentiment):
    text_class_list = []
    text_class_list.append(tweet.text)
    text_class_list.append(sentiment)
    writer.writerow(text_class_list)

def getTweets():
    loaded_model = pickle.load(open(filename, 'rb'))
    count_vect = pickle.load(open('count_vect', 'rb'))

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch('brexit')\
                                                .setTopTweets(True)\
                                                .setSince("2015-04-30")\
                                                .setUntil("2016-04-30")\
                                                .setMaxTweets(10000)
                                              
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    want_stay_count = 0
    want_quit_count = 0
    if len(tweets) > 0:
        for tweet in tweets:
            cleaned_tweet = data_cleaner.split_into_sentences(tweet.text)
            cleaned_tweet = [cleaned_tweet[0]]
            prediction = loaded_model.predict(count_vect.transform(cleaned_tweet))
            if prediction[0] == 0:
                want_quit_count+=1
                writeTweetsIntoCSV(tweet, "negative")
            else:
                want_stay_count+=1
                writeTweetsIntoCSV(tweet, "positive")
    f.close()
    draw_pie_chart(want_quit_count, want_stay_count)

def draw_pie_chart(want_to_exit, want_to_stay):
    labels = 'Want to Exit', 'Want to Stay'
    sizes = [want_to_exit, want_to_stay]
    colors = ['yellowgreen', 'lightcoral']
    explode = (0.1, 0)  # explode 1st slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()
if __name__ == "__main__":
    getTweets()
