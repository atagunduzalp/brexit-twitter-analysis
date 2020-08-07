import GetOldTweets3 as got
import csv

# f = open('want-to-exit.csv', 'w')
# writer = csv.writer(f)
# writer.writerow(['tweet', 'sentiment'])

f = open('want-to-exit.csv', 'w')
writer = csv.writer(f)
writer.writerow(['tweet', 'sentiment'])

def writeTweetsIntoCSV(tweet, sentiment):

    text_class_list = []
    text_class_list.append(tweet.text)
    text_class_list.append(sentiment)
    writer.writerow(text_class_list)
    # print("Supportive Tweets: " + tweet)

def getTweets(username_list, sentiment):
    
    for username in username_list:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch('brexit')\
                                                .setTopTweets(True)\
                                                .setSince("2015-04-30")\
                                                .setUntil("2017-04-30")\
                                                .setMaxTweets(1000)\
                                                .setUsername(username)
                                              
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        if len(tweets) > 0:
            for tweet in tweets:
                writeTweetsIntoCSV(tweet, sentiment)