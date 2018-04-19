import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz

dataset = pd.read_csv('network_backup_dataset.csv')
feature_names = ['Week #', 'Day of Week', 'Backup Start Time - Hour of Day', 'Work-Flow-ID','File Name']
class_names = ['Size of Backup (GB)']

# pre-processing dataset
dataset = dataset.replace({'Work-Flow-ID': {'work_flow_0': 0, 'work_flow_1': 1, 'work_flow_2': 2, 'work_flow_3': 3, 'work_flow_4': 4}})
dataset = dataset.replace({'Day of Week': {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3 , 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}})

file_name = []
for fname in dataset['File Name']:
    file_name.append(fname[5:])
dataset.drop(['File Name'], axis = 1, inplace = True)
dataset.insert(4, 'File Name', file_name)
dataset.drop(['Backup Time (hour)'], axis = 1, inplace = True)

# question i
data_true = dataset['Size of Backup (GB)']

rfr = RandomForestRegressor(n_estimators = 20, max_features = 5, max_depth = 4, bootstrap = True, random_state = 0, oob_score = True)
kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
rmse_train = 0
rmse_test = 0
for trainset, testset in kf.split(dataset):
    trainset_val = dataset.iloc[trainset]
    testset_val = dataset.iloc[testset]
    train_true = trainset_val['Size of Backup (GB)']
    test_true = testset_val['Size of Backup (GB)']
    trainset_val.drop(['Size of Backup (GB)'], axis = 1, inplace = True)
    testset_val.drop(['Size of Backup (GB)'], axis = 1, inplace = True)
    rfr.fit(trainset_val, train_true)
    rfr_pred_train = rfr.predict(trainset_val)
    rfr_pred_test = rfr.predict(testset_val)
    rmse_train += mean_squared_error(train_true, rfr_pred_train)
    rmse_test += mean_squared_error(test_true, rfr_pred_test)
print ('Out of Bag error: ' + str(1-rfr.oob_score_))
print ('average rmse for trainset: ' + str(np.sqrt(rmse_train/10)))
print ('average rmse for testset: ' + str(np.sqrt(rmse_test/10)))

# question ii
dataset_2 = dataset.copy()
data_true = dataset_2.pop('Size of Backup (GB)')
tree_num = np.arange(1, 201)
max_feature_num = [1, 2, 3, 4, 5]

def plot_avg_rmse(error):
    plt.plot(rmse[0], label = 'max_feature_num = 1')
    plt.plot(rmse[1], label = 'max_feature_num = 2')
    plt.plot(rmse[2], label = 'max_feature_num = 3')
    plt.plot(rmse[3], label = 'max_feature_num = 4')
    plt.plot(rmse[4], label = 'max_feature_num = 5')
    plt.title('average Test-RMSE against # trees')
    plt.xlabel('number of trees')
    plt.ylabel('average Test-RMSE')
    plt.legend(loc = 'best')
    plt.show()

def plot_oob_error(error):
    plt.plot(oob_error[0], label = 'max_feature_num = 1')
    plt.plot(oob_error[1], label = 'max_feature_num = 2')
    plt.plot(oob_error[2], label = 'max_feature_num = 3')
    plt.plot(oob_error[3], label = 'max_feature_num = 4')
    plt.plot(oob_error[4], label = 'max_feature_num = 5')
    plt.title('oob_error against # trees')
    plt.xlabel('number of trees')
    plt.ylabel('oob_error')
    plt.legend(loc = 'best')
    plt.show()

rmse = [[], [], [], [], []]
oob_error = [[], [], [], [], []]

for i in tree_num:
    for j in max_feature_num:
        rfr = RandomForestRegressor(n_estimators = i, max_features = j, max_depth = 4, bootstrap = True, oob_score = True, random_state = 0)
        rfr.fit(dataset_2, data_true)
        rfr_pred = cross_val_predict(rfr, dataset_2, data_true, cv = 10)
        rmse[j-1].append(np.sqrt(mean_squared_error(data_true, rfr_pred)))
        oob_error[j-1].append(1-rfr.oob_score_)
print (rmse)
print (oob_error)
plot_avg_rmse(rmse)
plot_oob_error(rmse)

# question iii
# plot avg rmse against max_depth
dataset_3 = dataset.copy()
data_true = dataset_3.pop('Size of Backup (GB)')
max_depth = np.arange(1, 21)
max_feature_num = [1, 2, 3, 4, 5] 
print ('finding best max_depth...')
for i in max_feature_num:
    rmse = []
    for j in max_depth:
        rfr = RandomForestRegressor(n_estimators = 12, max_features = i, max_depth = j, bootstrap = True, random_state = 0, oob_score = True)
        rfr.fit(dataset_3, data_true)
        rfr_pred = cross_val_predict(rfr, dataset_3, data_true, cv = 10)
        rmse.append(np.sqrt(mean_squared_error(data_true, rfr_pred)))
    print (rmse)
    plt.plot(rmse, label = 'max_feature_num = %d' % i)
plt.title('average Test-RMSE against max_depth')
plt.xlabel('max_depth')
plt.ylabel('average Test-RMSE')
plt.legend(loc = 'best')
plt.show()

# plot oob error against max_depth
for i in max_feature_num:
    oob_error = []
    for j in max_depth:
        rfr = RandomForestRegressor(n_estimators = 12, max_features = i, max_depth = j, bootstrap = True, random_state = 0, oob_score = True)
        rfr.fit(dataset_3, data_true)
        rfr_pred = cross_val_predict(rfr, dataset_3, data_true, cv = 10)
        oob_error.append(1-rfr.oob_score_)
    print (oob_error)
    plt.plot(oob_error, label = 'max_feature_num = %d' % i)
plt.title('average out_of_bag_error against max_depth')
plt.xlabel('max_depth')
plt.ylabel('out of bag error')
plt.legend(loc = 'best')  
plt.show()

# question iv
rfr = RandomForestRegressor(n_estimators = 20, max_features = 4, max_depth = 4, bootstrap = True, random_state = 0, oob_score = True)

print ('evaluate importance of Week # ...')
training_data = dataset.copy()
training_data.drop(['Week #'], axis = 1, inplace = True)
size_true = training_data.pop('Size of Backup (GB)')
rfr.fit(training_data, size_true)
size_pred = cross_val_predict(rfr, training_data, size_true, cv = 10)
print ('avg rmse when ignoring Week # :' + str(np.sqrt(mean_squared_error(size_true, size_pred))))
print ('oob_error when ignoring Week # :' + str(1-rfr.oob_score_))
print ('=' * 50)

print ('evaluate importance of Day of Week ...')
training_data = dataset.copy()
training_data.drop(['Day of Week'], axis = 1, inplace = True)
size_true = training_data.pop('Size of Backup (GB)')
rfr.fit(training_data, size_true)
size_pred = cross_val_predict(rfr, training_data, size_true, cv = 10)
print ('avg rmse when ignoring Day of Week :' + str(np.sqrt(mean_squared_error(size_true, size_pred))))
print ('oob_error when ignoring Day of Week :' + str(1-rfr.oob_score_))
print ('=' * 50)

print ('evaluate importance of Backup Start Time - Hour of Day ...')
training_data = dataset.copy()
training_data.drop(['Backup Start Time - Hour of Day'], axis = 1, inplace = True)
size_true = training_data.pop('Size of Backup (GB)')
rfr.fit(training_data, size_true)
size_pred = cross_val_predict(rfr, training_data, size_true, cv = 10)
print ('avg rmse when ignoring Backup Start Time - Hour of Day :' + str(np.sqrt(mean_squared_error(size_true, size_pred))))
print ('oob_error when ignoring Backup Start Time - Hour of Day :' + str(1-rfr.oob_score_))
print ('=' * 50)

print ('evaluate importance of Work-Flow-ID ...')
training_data = dataset.copy()
training_data.drop(['Work-Flow-ID'], axis = 1, inplace = True)
size_true = training_data.pop('Size of Backup (GB)')
rfr.fit(training_data, size_true)
size_pred = cross_val_predict(rfr, training_data, size_true, cv = 10)
print ('avg rmse when ignoring Work-Flow-ID :' + str(np.sqrt(mean_squared_error(size_true, size_pred))))
print ('oob_error when ignoring Work-Flow-ID :' + str(1-rfr.oob_score_))
print ('=' * 50)

print ('evaluate importance of File Name ...')
training_data = dataset.copy()
training_data.drop(['File Name'], axis = 1, inplace = True)
size_true = training_data.pop('Size of Backup (GB)')
rfr.fit(training_data, size_true)
size_pred = cross_val_predict(rfr, training_data, size_true, cv = 10)
print ('avg rmse when ignoring File Name :' + str(np.sqrt(mean_squared_error(size_true, size_pred))))
print ('oob_error when ignoring File Name :' + str(1-rfr.oob_score_))
print ('=' * 50)

# question v
dataset_5 = dataset.copy()
data_true = dataset_5.pop('Size of Backup (GB)')

rfr = RandomForestRegressor(n_estimators = 12, max_depth = 4, max_features = 3)
rfr.fit(dataset_5, data_true)
print ('feature importance' + str(rfr.feature_importances_))
print (rfr.estimators_[5] )
dot_data = tree.export_graphviz(rfr.estimators_[5], out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dataset")

dot_data = tree.export_graphviz(rfr.estimators_[5], out_file=None, 
                         feature_names=feature_names,  
                         class_names=class_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.view()

# plot Fitted values and Actual values vs. dataset index
dataset_copy = dataset.copy()
data_true = dataset_copy.pop('Size of Backup (GB)')
rfr = RandomForestRegressor(n_estimators = 12, max_depth = 9, max_features = 3)
rfr.fit(dataset_copy, data_true)
data_pred = cross_val_predict(rfr, dataset_copy, data_true, cv = 10)
print ('avg rmse with best parameters :' + str(np.sqrt(mean_squared_error(data_true, data_pred))))

plt.scatter(dataset_copy.index, data_true, label = 'actual values')
plt.scatter(dataset_copy.index, data_pred, label = 'fitted values')
plt.xlabel('index of data points', fontsize = 20)
plt.ylabel('backup size', fontsize = 20)
plt.title('Fitted values vs. Actual values', fontsize = 20)
plt.legend(loc = 'best')
plt.show()

# plot Residuals and Fitted values vs. dataset index
plt.scatter(dataset_copy.index, data_pred - data_true, label = 'residuals')
plt.scatter(dataset_copy.index, data_pred, label = 'fitted values')
plt.xlabel('index of data points', fontsize = 20)
plt.ylabel('residuals', fontsize = 20)
plt.title('Residuals vs. Fitted value', fontsize = 20)
plt.legend(loc = 'best')
plt.show()
