import json
import numpy as np
import datetime, time
import pytz
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import os


hashtags = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt', 'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']
mt_tz = pytz.timezone('US/Pacific')
time1 = 1422806400
time2 = 1422849600

def kfold(x, y = None, k = 10):
    # creating folds
    # x is an numpy 2d array, y is a list having the same length as number of x's rows
    x_folds = []
    l = len(x)
    for i in range(k):
        x_folds.append(x[int(i*l/k):int((i+1)*l/k)])
    if y == None:
        return x_folds
    else:
        y_folds = []
        for i in range(k):
            y_folds.append(y[int(i*l/k):int((i+1)*l/k)])
    
        return x_folds, y_folds
    
def cv(x_folds, y_folds, func):
    num_folds = len(x_folds)
    error = 0
    for i in range(num_folds):
        # building training set and validation set from folds
        x_v = x_folds[i]
        y_v = np.asarray(y_folds[i])
        y_t = []
        if i == 0:
            x_t = x_folds[1]
            y_t.extend(y_folds[1])
            for j in range(2, num_folds):
                x_t = np.vstack((x_t, x_folds[j]))
                y_t.extend(y_folds[j])
        else:
            x_t = x_folds[0]
            y_t.extend(y_folds[0])
            for j in range(1, num_folds):
                if j != i:
                    x_t = np.vstack((x_t, x_folds[j]))
                    y_t.extend(y_folds[j])
        # fitting and predicting
        func.fit(x_t, y_t)
        pred_v = func.predict(x_v)
        error += np.mean(np.abs(pred_v-y_v))
    return error/num_folds
    
# extracting features
def extracting(hashtag):
    file = open('./tweet_data/'+hashtag, encoding = 'utf8')
    posting_time = []
    num_retweets = []
    num_followers = []
    num_URLs = []
    authors = []
    num_mentions = []
    ranking_score = []
    num_hashtags = []
    for line in file:
        if line != '\n':
            data = json.loads(line)
            posting_time.append(data['citation_date'])
            num_retweets.append(data['metrics']['citations']['total'])
            num_followers.append(data['author']['followers'])
            num_URLs.append(len(data['tweet']['entities']['urls']))
            authors.append(data['author']['nick'])
            num_mentions.append(len(data['tweet']['entities']['user_mentions']))
            ranking_score.append(data['metrics']['ranking_score'])
            num_hashtags.append(data['title'].count('#'))
    file.close()
    start_time = min(posting_time)
    end_time = max(posting_time)
    hours1 = int(np.ceil((time1-start_time)/3600))
    hours2 = 12
    hours3 = int(np.ceil((end_time-time2)/3600))
    epoch1 = np.zeros([hours1, 4])
    epoch2 = np.zeros([hours2, 4])
    epoch3 = np.zeros([hours3, 4])
    unique_authors1 = [[] for i in range(hours1)]
    unique_authors2 = [[] for i in range(hours2)]
    unique_authors3 = [[] for i in range(hours3)]
    for i in range(len(posting_time)):
        if posting_time[i] < time1:
            row = hours1-int(np.ceil((time1-posting_time[i])/3600))
            epoch = epoch1
            unique_authors = unique_authors1
        elif posting_time[i] > time2:
            row = int(np.ceil((posting_time[i]-time2)/3600))-1
            epoch = epoch3
            unique_authors = unique_authors3
        else:
            row = int(np.ceil((posting_time[i]-time1)/3600))-1
            epoch = epoch2
            unique_authors = unique_authors2
        if hashtag == 'tweets_#gohawks.txt':
            epoch[row, 0] += ranking_score[i]
            epoch[row, 1] += 1
            epoch[row, 2] += num_mentions[i]
            epoch[row, 3] += 1
        elif hashtag == 'tweets_#gopatriots.txt':
            epoch[row, 0] += num_retweets[i]
            epoch[row, 1] += num_URLs[i]
            if authors[i] not in unique_authors[row]:
                unique_authors[row].append(authors[i])
                epoch[row, 2] += 1
            epoch[row, 3] += 1
        elif hashtag == 'tweets_#nfl.txt':
            epoch[row, 0] += num_hashtags[i]
            if authors[i] not in unique_authors[row]:
                unique_authors[row].append(authors[i])
                epoch[row, 1] += 1
            epoch[row, 2] += num_mentions[i]
            epoch[row, 3] += 1
        elif hashtag == 'tweets_#patriots.txt':
            epoch[row, 0] += 1
            epoch[row, 1] += ranking_score[i]
            epoch[row, 2] += num_retweets[i]
            epoch[row, 3] += 1
        elif hashtag == 'tweets_#sb49.txt':
            epoch[row, 0] += num_followers[i]
            epoch[row, 1] += 1
            epoch[row, 2] += ranking_score[i]
            epoch[row, 3] += 1
        elif hashtag == 'tweets_#superbowl.txt':
            epoch[row, 0] += 1
            epoch[row, 1] += ranking_score[i]
            if authors[i] not in unique_authors[row]:
                unique_authors[row].append(authors[i])
                epoch[row, 2] += 1            
            epoch[row, 3] += 1
        else:
            epoch[row, 0] += 1
            epoch[row, 1] += ranking_score[i]
            epoch[row, 2] += num_followers[i]      
            epoch[row, 3] += 1
    return epoch1, epoch2, epoch3
    

def fitting(all):
    np.random.seed(1)
    np.random.shuffle(all)
    data = all[:-1, :-1]
    target = all[1:, 3].reshape(1, data.shape[0]).tolist()[0]
    data_folds, target_folds = kfold(data, target)
    
    lr = linear_model.LinearRegression()
    cv_error_lr = cv(data_folds, target_folds, lr)
    rfr = RandomForestRegressor(n_estimators = 50, random_state = 0)
    cv_error_rfr = cv(data_folds, target_folds, rfr)
    knn = KNeighborsRegressor(n_neighbors = 9)
    cv_error_knn = cv(data_folds, target_folds, knn)
    
    return cv_error_lr, cv_error_rfr, cv_error_knn
    
# 4_1
for hashtag in hashtags:
    epoch1, epoch2, epoch3 = extracting(hashtag)
    cv_error_lr, cv_error_rfr, cv_error_knn = fitting(epoch1)
    print ('For hashtag '+hashtag[7:-4]+':')
    print ('In epoch1, before Feb. 1, 8:00 a.m., the average cross-validation error(MAE) using linear model is '+str(cv_error_lr))
    print ('In epoch1, before Feb. 1, 8:00 a.m., the average cross-validation error(MAE) using random forest is '+str(cv_error_rfr))
    print ('In epoch1, before Feb. 1, 8:00 a.m., the average cross-validation error(MAE) using knn is '+str(cv_error_knn))
    cv_error_lr, cv_error_rfr, cv_error_knn = fitting(epoch2)
    print ('In epoch2, between Feb. 1, 8:00 a.m. and 8:00 p.m., the average cross-validation error(MAE) using linear model is '+str(cv_error_lr))
    print ('In epoch2, between Feb. 1, 8:00 a.m. and 8:00 p.m., the average cross-validation error(MAE) using random forest is '+str(cv_error_rfr))
    print ('In epoch2, between Feb. 1, 8:00 a.m. and 8:00 p.m., the average cross-validation error(MAE) using knn is '+str(cv_error_knn))
    cv_error_lr, cv_error_rfr, cv_error_knn = fitting(epoch3)
    print ('In epoch3, after Feb. 1, 8:00 p.m., the average cross-validation error(MAE) using linear model is '+str(cv_error_lr))
    print ('In epoch3, after Feb. 1, 8:00 p.m., the average cross-validation error(MAE) using random forest is '+str(cv_error_rfr))
    print ('In epoch3, after Feb. 1, 8:00 p.m., the average cross-validation error(MAE) using knn is is '+str(cv_error_knn))
    print ('-'*20)
print ('-'*30)

# 4_2
# meragefiledir = os.getcwd()+'\\test_data'
# filenames = os.listdir(meragefiledir)
# file = open('./test_data/aggregated data.txt','w')
# for filename in filenames:    
    # filepath = meragefiledir+'\\'+filename      
    # for line in open(filepath):    
        # file.writelines(line)
    # file.write('\n')        
# file.close()

# epoch1, epoch2, epoch3 = extracting('aggregated data.txt')
# cv_error_lr, cv_error_rfr, cv_error_knn = fitting(epoch1)
# print ('For aggregated data:')
# print ('In epoch1, before Feb. 1, 8:00 a.m., the average cross-validation error(MAE) using linear model is '+str(cv_error_lr))
# print ('In epoch1, before Feb. 1, 8:00 a.m., the average cross-validation error(MAE) using random forest is '+str(cv_error_rfr))
# print ('In epoch1, before Feb. 1, 8:00 a.m., the average cross-validation error(MAE) using knn is '+str(cv_error_knn))
# cv_error_lr, cv_error_rfr, cv_error_knn = fitting(epoch2)
# print ('In epoch2, between Feb. 1, 8:00 a.m. and 8:00 p.m., the average cross-validation error(MAE) using linear model is '+str(cv_error_lr))
# print ('In epoch2, between Feb. 1, 8:00 a.m. and 8:00 p.m., the average cross-validation error(MAE) using random forest is '+str(cv_error_rfr))
# print ('In epoch2, between Feb. 1, 8:00 a.m. and 8:00 p.m., the average cross-validation error(MAE) using knn is '+str(cv_error_knn))
# cv_error_lr, cv_error_rfr, cv_error_knn = fitting(epoch3)
# print ('In epoch3, after Feb. 1, 8:00 p.m., the average cross-validation error(MAE) using linear model is '+str(cv_error_lr))
# print ('In epoch3, after Feb. 1, 8:00 p.m., the average cross-validation error(MAE) using random forest is '+str(cv_error_rfr))
# print ('In epoch3, after Feb. 1, 8:00 p.m., the average cross-validation error(MAE) using knn is is '+str(cv_error_knn))