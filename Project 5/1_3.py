import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt

tweet_file = ['#gohawks', '#gopatriots', '#nfl', '#patriots', '#sb49', '#superbowl']
features_3 = ['tweets_num', 'retweets_num', 'sum_followers', 'max_followers', 'URLs_num', 
              'authors_num', 'mentions_num', 'ranking_score', 'hashtags_num']

def plot_feature(fea_val, pred, hashtag, feature_name):
    plt.scatter(fea_val, pred, color = 'blue')
    plt.xlabel(feature_name)
    plt.ylabel('predictant')
    plt.title('# tweet for next hour vs. ' + feature_name + ' (tweets_' + hashtag + ')')
    plt.grid(True)
    plt.savefig('q1.3_' + hashtag + '_' + feature_name + '.png')

for hashtag in tweet_file:
    # extract_feature
    label = {'#gohawks' : ['tweets_#gohawks.txt', 188136], '#gopatriots' : ['tweets_#gopatriots.txt', 26232],
             '#nfl' : ['tweets_#nfl.txt', 259024], '#patriots' : ['tweets_#patriots.txt', 489713],
             '#sb49' : ['tweets_#sb49.txt', 826951], '#superbowl' : ['tweets_#superbowl.txt', 1348767]}
    time_stamps, author_names, user_followers, retweet, url_citation_num, mention_num, ranking_scores, hashtag_num = [],[],[],[],[],[],[],[]

    input_file = open('./tweet_data/' + label[hashtag][0], encoding = 'utf-8')
    for (line, index) in zip(input_file, range(0, label[hashtag][1])):
        data = json.loads(line)
        time_stamps.append(data['citation_date'])
        author_name = data['author']['nick']
        original_author_name = data['original_author']['nick']
        user_followers.append(data['author']['followers'])
        if author_name != original_author_name:
            retweet.append(1)
        else:
            retweet.append(0)
        url_citation_num.append(len(data['tweet']['entities']['urls']))
        author_names.append(author_name)
        mention_num.append(len(data['tweet']['entities']['user_mentions']))
        ranking_scores.append(data['metrics']['ranking_score'])
        hashtag_num.append(data['title'].count('#'))
    input_file.close()

    prev_hour = int((max(time_stamps)-min(time_stamps))/3600)+1
    hour_tweet_num = [0] * prev_hour
    hour_retweet_num = [0] * prev_hour
    hour_follower_sum = [0] * prev_hour
    max_hour_followers_num = [0] * prev_hour
    hour_time_of_the_day = [0] * prev_hour
    hour_url_citation_num = [0] * prev_hour
    hour_author_num = [0] * prev_hour
    hour_author_set = [0] * prev_hour
    for i in range(0, prev_hour):
        hour_author_set[i] = set([])
    hour_mention_num = [0] * prev_hour
    hour_ranking_scores_tot = [0.0] * prev_hour
    hour_hashtag_num = [0] * prev_hour
    
    start_time = min(time_stamps)
    for i in range(0, len(hour_time_of_the_day)):
        hour_time_of_the_day[i] = i % 24
    for i in range(0, label[hashtag][1]):
        pres_hour = int((time_stamps[i]-start_time)/3600)
        hour_tweet_num[pres_hour] += 1
        if retweet[i] == 1:
            hour_retweet_num[pres_hour] += 1
        hour_follower_sum[pres_hour] += user_followers[i]
        if user_followers[i] > max_hour_followers_num[pres_hour]:
            max_hour_followers_num[pres_hour] = user_followers[i]
        hour_url_citation_num[pres_hour] += url_citation_num[i]
        hour_author_set[pres_hour].add(author_names[i])
        hour_mention_num[pres_hour] += mention_num[i]
        hour_ranking_scores_tot[pres_hour] += ranking_scores[i]
        hour_hashtag_num[pres_hour] += hashtag_num[i]
    for i in range(0, len(hour_author_set)):
        hour_author_num[i] = len(hour_author_set[i])
    
    target_value = hour_tweet_num[1:]
    target_value.append(0)
    data = np.array([hour_tweet_num, hour_retweet_num, hour_follower_sum, max_hour_followers_num,hour_time_of_the_day, 
                     hour_url_citation_num, hour_author_num,hour_mention_num, hour_ranking_scores_tot, hour_hashtag_num, target_value])
    data = np.transpose(data)
    df = DataFrame(data)
    df.columns = ['tweets_num', 'retweets_num', 'sum_followers', 'max_followers', 'time_of_day',
                  'URLs_num', 'authors_num', 'mentions_num', 'ranking_score', 'hashtags_num', 'target_value']
    training_data = df
    
    # one-hot encoding
    time_of_day_set = range(0,24)
    for time_of_day in time_of_day_set:
        time_of_day_to_add = []
        for time_of_day_item in training_data['time_of_day']:
            if time_of_day_item == time_of_day:
                time_of_day_to_add.append(1)
            else:
                time_of_day_to_add.append(0)
        training_data.insert(training_data.shape[1]-1, str(time_of_day)+'th_hour', time_of_day_to_add)
    
    # linear regression
    training_data.drop('time_of_day', axis = 1, inplace = True)
    target_data = training_data.pop('target_value')
    
    lr = LinearRegression()
    lr_result = lr.fit(training_data, target_data)
    lr_pred = lr.predict(training_data)
    print ('rmse for tweets_' +hashtag+ ': ' + str(np.sqrt(mean_squared_error(target_data, lr_pred))))
    
    # perform t-test
    model = sm.OLS(target_data, training_data)
    result = model.fit()
    print (result.summary())
    p_val = result.pvalues[0:9]
    print ('P-values for each feature of tweets_' + hashtag + ' are: ')
    print (p_val)
    index = sorted(range(len(p_val)), key = lambda i: p_val[i])[0:3]
    print('top 3 features are:')
    print(features_3[index[0]], features_3[index[1]], features_3[index[2]])
    plot_feature(training_data[features_3[index[0]]], lr_pred, hashtag, features_3[index[0]])
    plot_feature(training_data[features_3[index[1]]], lr_pred, hashtag, features_3[index[1]])
    plot_feature(training_data[features_3[index[2]]], lr_pred, hashtag, features_3[index[2]])
    print ('='*50)
