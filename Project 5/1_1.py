import json
import matplotlib.pyplot as plt

def plot_histogram(posting_time, hashtag):
    hours = int((max(posting_time)-min(posting_time))/3600)
    start_time = min(posting_time)
    for i in range(len(posting_time)):
        posting_time[i] = (posting_time[i]-start_time)/3600
    plt.hist(posting_time, bins = hours)
    plt.xlabel('time(hour)')
    plt.ylabel('number of tweets')
    plt.title('number of tweets in hour over time for hashtag '+hashtag)
    plt.show()       

hashtags = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt', 'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']

for hashtag in hashtags:
    file = open('./tweet_data/'+hashtag, encoding = 'utf8')
    posting_time = []
    followers = 0
    retweets = 0    
    for line in file:
        data = json.loads(line)
        posting_time.append(data['citation_date'])
        followers += data['author']['followers']
        retweets += data['metrics']['citations']['total']
    file.close()
    num_of_tweets = len(posting_time)
    avg_followers = followers/num_of_tweets
    avg_retweets = retweets/num_of_tweets
    hours = (max(posting_time)-min(posting_time))/3600
    avg_tweets = num_of_tweets/hours
    
    print ('For hashtag '+hashtag[7:-4]+':')
    print ('Average number of tweets per hour is '+str(avg_tweets))
    print ('Average number of followers of users posting the tweets per hour is '+str(avg_followers))
    print ('Average number of retweets per hour is '+str(avg_retweets))
    print ('-'*20)
    
    if hashtag == 'tweets_#nfl.txt':
        plot_histogram(posting_time, '#NFL')
    if hashtag == 'tweets_#superbowl.txt':
        plot_histogram(posting_time, '#SuperBowl')