import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor

hashtags = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt', 'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']
month = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct' : '10', 'Nov':'11', 'Dec':'12'}

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
    num_followers = []
    num_friends = []
    num_favorite = []
    num_lists = []
    creating_time = []
    description = []
    default_profile = []
    default_profile_image = []
    total_tweet = []
    for line in file:
        data = json.loads(line)
        num_followers.append(data['tweet']['user']['followers_count'])
        num_friends.append(data['tweet']['user']['friends_count'])
        num_favorite.append(data['tweet']['user']['favourites_count'])
        if data['tweet']['user']['listed_count']:
            num_lists.append(data['tweet']['user']['listed_count'])
        else:
            num_lists.append(0)
        tmp = data['tweet']['user']['created_at'][-4:]+month[data['tweet']['user']['created_at'][4:7]]+data['tweet']['user']['created_at'][8:10]
        creating_time.append(int(tmp))
        if data['tweet']['user']['description']:
            description.append(1)
        else:
            description.append(0)
        if data['tweet']['user']['default_profile'] == True:
            default_profile.append(1)
        else:
            default_profile.append(0)
        if data['tweet']['user']['default_profile_image'] == True:
            default_profile_image.append(1)
        else:
            default_profile_image.append(0)
        total_tweet.append(data['tweet']['user']['statuses_count'])
    file.close()
    all = (np.vstack((np.asarray(num_followers), np.asarray(num_friends), np.asarray(num_favorite), np.asarray(num_lists), np.asarray(creating_time), 
           np.asarray(description), np.asarray(default_profile), np.asarray(default_profile_image), np.asarray(total_tweet)))).T    
    return all
    
def fitting(all):
    np.random.seed(1)
    np.random.shuffle(all)
    data = all[:, :-1]
    target = all[:, -1].reshape(1, data.shape[0]).tolist()[0]
    data_folds, target_folds = kfold(data, target)
    
    rfr = RandomForestRegressor(n_estimators = 50, random_state = 0)
    cv_error = cv(data_folds, target_folds, rfr)
    
    return cv_error, np.mean(all[:, -1])
    
for hashtag in hashtags:
    all = extracting(hashtag)
    cv_error, avg = fitting(all)
    print ('For hashtag '+hashtag[7:-4]+':')
    print ('The average cross-validation error(MAE) using random forest is '+str(cv_error))
    print ('The average tweet per user is '+str(avg))
    print ('-'*20)
print ('-'*30)