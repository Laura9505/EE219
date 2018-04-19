import json
import numpy as np
import statsmodels.api as sm
import datetime, time
import pytz
from sklearn import linear_model

hashtags = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt', 'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']
mt_tz = pytz.timezone('US/Mountain')
    
# extracting features
def extracting(hashtag):
    file = open('./tweet_data/'+hashtag, encoding = 'utf8')
    posting_time = []
    num_retweets = []
    num_followers = []
    for line in file:
        data = json.loads(line)
        posting_time.append(data['citation_date'])
        num_retweets.append(data['metrics']['citations']['total'])
        num_followers.append(data['author']['followers'])
    file.close()
    hours = int((max(posting_time)-min(posting_time))/3600)+1
    tmp = np.zeros([hours, 5])
    start_time = min(posting_time)
    start_hour = (datetime.datetime.fromtimestamp(start_time, mt_tz)).hour
    for i in range(hours):
        tmp[i,4] = (start_hour+i)%24
    for i in range(len(posting_time)):
        tmp[int((posting_time[i]-start_time)/3600), 0] += 1
        tmp[int((posting_time[i]-start_time)/3600), 1] += num_retweets[i]
        tmp[int((posting_time[i]-start_time)/3600), 2] += num_followers[i]
        if tmp[int((posting_time[i]-start_time)/3600), 3] < num_followers[i]:
            tmp[int((posting_time[i]-start_time)/3600), 3] = num_followers[i]
    return tmp
    
def feature_analysis(all):
    data = all[:-1, :]
    target = all[1:, 0]
    model = sm.OLS(target, data)
    results = model.fit()
    print (results.summary())

def fitting(all):
    np.random.seed(1)
    np.random.shuffle(all)
    data = all[:-1, :]
    target = all[1:, 0].reshape(1, data.shape[0]).tolist()[0]
    
    lr = linear_model.LinearRegression()
    lr.fit(data, target)
    pred = lr.predict(data)
    error = np.mean(np.abs(pred-all[1:, 0]))
    
    return error

for hashtag in hashtags:
    all = extracting(hashtag)
    print ('For hashtag '+hashtag[7:-4]+':')
    feature_analysis(all)
    error = fitting(all)    
    print ('The training error(MAE) is '+str(error))
    print ('-'*20)