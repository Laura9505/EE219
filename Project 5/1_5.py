import os
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
'''
using K-nn Regressor, using features: tweet_number, ranking_score, user_followers
'''

# extract train data
train_len = 3138823
time_stamps, user_followers, ranking_scores = [],[],[]

input_file = open('./tweet_data/train_merge.txt', encoding = 'utf-8')
for (line, index) in zip(input_file, range(0, train_len)):
    data = json.loads(line)
    time_stamps.append(data['citation_date'])
    user_followers.append(data['author']['followers'])
    ranking_scores.append(data['metrics']['ranking_score'])
input_file.close()
print ('extract data from train file done!')

start_time = 1421222400
prev_hour = int((max(time_stamps)-min(time_stamps))/3600)+1
hour_tweet_num = [0] * prev_hour
hour_follower_sum = [0] * prev_hour
max_hour_followers_num = [0] * prev_hour
hour_ranking_scores_tot = [0.0] * prev_hour

for i in range(0, train_len):
    pres_hour = int((time_stamps[i]-start_time)/3600)
    hour_tweet_num[pres_hour] += 1
    hour_follower_sum[pres_hour] += user_followers[i]
    if user_followers[i] > max_hour_followers_num[pres_hour]:
        max_hour_followers_num[pres_hour] = user_followers[i]
    hour_ranking_scores_tot[pres_hour] += ranking_scores[i]

target_value = hour_tweet_num[5:]
target_value = target_value + [0, 0, 0, 0, 0]
data = np.array([hour_tweet_num, hour_follower_sum, hour_ranking_scores_tot, target_value])
data = np.transpose(data)
df = DataFrame(data)
df.columns = ['tweets_num_0', 'sum_followers_0', 'ranking_score_0', 'target_value']

df['tweets_num_1'] = [0.0]*len(df)
df['sum_followers_1'] = [0.0]*len(df)
df['ranking_score_1'] = [0.0]*len(df)

df['tweets_num_2'] = [0.0]*len(df)
df['sum_followers_2'] = [0.0]*len(df)
df['ranking_score_2'] = [0.0]*len(df)

df['tweets_num_3'] = [0.0]*len(df)
df['sum_followers_3'] = [0.0]*len(df)
df['ranking_score_3'] = [0.0]*len(df)

df['tweets_num_4'] = [0.0]*len(df)
df['sum_followers_4'] = [0.0]*len(df)
df['ranking_score_4'] = [0.0]*len(df)
    
for i in range(0, len(df)-1):
    df['tweets_num_1'][i] = df['tweets_num_0'][i+1]
    df['sum_followers_1'][i] = df['sum_followers_0'][i+1]
    df['ranking_score_1'][i] = df['ranking_score_0'][i+1]
for i in range(0, len(df)-2):
    df['tweets_num_2'][i] = df['tweets_num_0'][i+2]
    df['sum_followers_2'][i] = df['sum_followers_0'][i+2]
    df['ranking_score_2'][i] = df['ranking_score_0'][i+2]
for i in range(0, len(df)-3):
    df['tweets_num_3'][i] = df['tweets_num_0'][i+3]
    df['sum_followers_3'][i] = df['sum_followers_0'][i+3]
    df['ranking_score_3'][i] = df['ranking_score_0'][i+3]
for i in range(0, len(df)-4):
    df['tweets_num_4'][i] = df['tweets_num_0'][i+4]
    df['sum_followers_4'][i] = df['sum_followers_0'][i+4]
    df['ranking_score_4'][i] = df['ranking_score_0'][i+4]

df.drop([i for i in range(len(df)-5,len(df))], inplace = True)

if os.path.isdir('./extracted_data'):
    pass
else:
    os.mkdir('./extracted_data')
df.to_csv('./extracted_data/q1.5_train_merge_v2.csv', index = False)


for test_data_index in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

    test_dict = {0 : ['sample1_period1.txt', 730],  1 : ['sample2_period2.txt', 212273], 2 : ['sample3_period3.txt', 3628], 
                 3 : ['sample4_period1.txt', 1646], 4 : ['sample5_period1.txt', 2059],   5 : ['sample6_period2.txt', 205554],
                 6 : ['sample7_period3.txt', 528],  7 : ['sample8_period1.txt', 229],    8 : ['sample9_period2.txt', 11311],
                 9 : ['sample10_period3.txt', 365]}
    time_stamps, user_followers, ranking_scores = [],[],[]

    input_file = open('./test_data/' + test_dict[test_data_index][0], encoding = 'utf-8')
    for (line, index) in zip(input_file, range(0, test_dict[test_data_index][1])):
        data = json.loads(line)
        time_stamps.append(data['firstpost_date'])
        user_followers.append(data['author']['followers'])
        ranking_scores.append(data['metrics']['ranking_score'])
    input_file.close()

    start_time = (min(time_stamps)/3600)*3600
    
    prev_hour = int((max(time_stamps)-min(time_stamps))/3600)+1
    hour_tweet_num = [0] * prev_hour
    hour_follower_sum = [0] * prev_hour
    max_hour_followers_num = [0] * prev_hour
    hour_ranking_scores_tot = [0.0] * prev_hour

    for i in range(0, test_dict[test_data_index][1]):
        pres_hour = int((time_stamps[i]-start_time)/3600)
        hour_tweet_num[pres_hour] += 1
        hour_follower_sum[pres_hour] += user_followers[i]
        if user_followers[i] > max_hour_followers_num[pres_hour]:
            max_hour_followers_num[pres_hour] = user_followers[i]
        hour_ranking_scores_tot[pres_hour] += ranking_scores[i]
    
    if test_data_index == 7:
        target_value = hour_tweet_num[4:]
        target_value = target_value + [0, 0, 0, 0]
    else:
        target_value = hour_tweet_num[5:]
        target_value = target_value + [0, 0, 0, 0, 0]

    data = np.array([hour_tweet_num, hour_follower_sum, hour_ranking_scores_tot, target_value])
    data = np.transpose(data)
    df = DataFrame(data)
    df.columns = ['tweets_num_0', 'sum_followers_0', 'ranking_score_0', 'target_value']
    
    # 5-hour time window
    df['tweets_num_1'] = [0.0]*len(df)
    df['sum_followers_1'] = [0.0]*len(df)
    df['ranking_score_1'] = [0.0]*len(df)

    df['tweets_num_2'] = [0.0]*len(df)
    df['sum_followers_2'] = [0.0]*len(df)
    df['ranking_score_2'] = [0.0]*len(df)

    df['tweets_num_3'] = [0.0]*len(df)
    df['sum_followers_3'] = [0.0]*len(df)
    df['ranking_score_3'] = [0.0]*len(df)

    df['tweets_num_4'] = [0.0]*len(df)
    df['sum_followers_4'] = [0.0]*len(df)
    df['ranking_score_4'] = [0.0]*len(df)
    
    for i in range(0, len(df)-1):
        df['tweets_num_1'][i] = df['tweets_num_0'][i+1]
        df['sum_followers_1'][i] = df['sum_followers_0'][i+1]
        df['ranking_score_1'][i] = df['ranking_score_0'][i+1]
    for i in range(0, len(df)-2):
        df['tweets_num_2'][i] = df['tweets_num_0'][i+2]
        df['sum_followers_2'][i] = df['sum_followers_0'][i+2]
        df['ranking_score_2'][i] = df['ranking_score_0'][i+2]
    for i in range(0, len(df)-3):
        df['tweets_num_3'][i] = df['tweets_num_0'][i+3]
        df['sum_followers_3'][i] = df['sum_followers_0'][i+3]
        df['ranking_score_3'][i] = df['ranking_score_0'][i+3]
    for i in range(0, len(df)-4):
        df['tweets_num_4'][i] = df['tweets_num_0'][i+4]
        df['sum_followers_4'][i] = df['sum_followers_0'][i+4]
        df['ranking_score_4'][i] = df['ranking_score_0'][i+4]

    if test_data_index == 7:
        df.drop([i for i in range(len(df)-4,len(df))], inplace = True)
    else:
        df.drop([i for i in range(len(df)-5,len(df))], inplace = True)

    train_data = pd.read_csv('./extracted_data/q1.5_train_merge_v2.csv')
    test_data = df

    # split data into three period
    train_target = train_data.pop('target_value')
    test_target = test_data.pop('target_value')
    
    train_data_p1 = train_data[:440]
    train_data_p2 = train_data[440:452]
    train_data_p3 = train_data[452:]
    train_target_p1 = train_target[:440]
    train_target_p2 = train_target[440:452]
    train_target_p3 = train_target[452:]
    
    knn_p1 = KNeighborsRegressor(n_neighbors = 50)
    knn_p2 = KNeighborsRegressor(n_neighbors = 12)
    knn_p3 = KNeighborsRegressor(n_neighbors = 50)
    knn_result_p1 = knn_p1.fit(train_data_p1, train_target_p1)
    knn_result_p2 = knn_p2.fit(train_data_p2, train_target_p2)
    knn_result_p3 = knn_p3.fit(train_data_p3, train_target_p3)

    knn_pred = []
    if test_dict[test_data_index][0][-5] == '1':
        knn_pred = knn_p1.predict(test_data)
    elif test_dict[test_data_index][0][-5] == '2':
        knn_pred = knn_p2.predict(test_data)
    else:
        knn_pred = knn_p3.predict(test_data)
    
    data = np.array([knn_pred, test_target])
    data = np.transpose(data)
    results = DataFrame(data)
    results.columns = ['Prediction', 'Actual']
    print (results)
    if test_data_index == 7:
        print ('no rmse for 6th hour!')
    else:
        print ('rmse for '+test_dict[test_data_index][0]+' : ' + str(np.sqrt(mean_squared_error(results['Prediction'], results['Actual']))))
