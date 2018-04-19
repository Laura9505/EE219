import csv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sklearn.metrics as metrics
from sklearn import linear_model
import sklearn.preprocessing as pre
import sklearn.feature_selection as fs

raw_data = []
with open('network_backup_dataset.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] != 'Week #':
            raw_data.append(row[:6])

dic = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}

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
    train_mse = 0
    test_mse = 0
    for i in range(num_folds):
        # building training set and validation set from folds
        x_v = x_folds[i]
        y_v = y_folds[i]
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
        pred_t = func.predict(x_t)
        train_mse += metrics.mean_squared_error(y_t, pred_t)
        pred_v = func.predict(x_v)
        test_mse += metrics.mean_squared_error(y_v, pred_v)
    return np.sqrt(train_mse/num_folds), np.sqrt(test_mse/num_folds)

def plot2(true, pred, title):
    tmp = len(true)
    x = list(range(tmp))
    residual = true - pred
    axes = plt.subplot(111)
    type1 = axes.scatter(x, true, c = 'red', marker = 'x')
    type2 = axes.scatter(x, pred, c = 'blue')
    plt.ylabel('fitted/true values')
    axes.legend((type1, type2), ('true', 'fitted'))
    plt.title('fitted values vs true values '+str(title))
    plt.show()
    axes = plt.subplot(111)
    type1 = axes.scatter(x, residual, c = 'red')
    type2 = axes.scatter(x, pred, c = 'blue')
    plt.ylabel('residual/fitted values')
    axes.legend((type1, type2), ('residual', 'fitted'))
    plt.title('residual vs fitted values '+str(title))
    plt.show()
    
# i
scalar = raw_data[:]
for i in range(len(scalar)):
    scalar[i][0] = int(scalar[i][0])
    scalar[i][1] = dic[scalar[i][1]]
    scalar[i][2] = int(scalar[i][2])
    scalar[i][3] = int(scalar[i][3][-1])
    scalar[i][4] = int(scalar[i][4][5:])
    scalar[i][5] = float(scalar[i][5])
   
lr = linear_model.LinearRegression()
tmp = np.asarray(scalar)
np.random.seed(1)
np.random.shuffle(tmp)
data = tmp[:, 0:5].astype(int)
target = tmp[:, 5].reshape(1, data.shape[0]).tolist()[0]

data_folds, target_folds = kfold(data, target)
train_rmse_basiclr, test_rmse_basiclr = cv(data_folds, target_folds, lr)
print ('For basic linear regression model with scalar encoding, the average training RMSE is '+str(train_rmse_basiclr)+', average test RMSE is '+str(test_rmse_basiclr))
lr.fit(data, target)
pred_basiclr = lr.predict(data)
plot2(target, pred_basiclr, 'using basic linear regression')
print ('-'*30)

# ii
data_std = pre.scale(data, axis = 1)
data_std_folds = kfold(data_std)
train_rmse_basiclr_std, test_rmse_basiclr_std = cv(data_std_folds, target_folds, lr)
print ('For basic linear regression model with Standardized scalar encoding, the average training RMSE is '+str(train_rmse_basiclr_std)+', average test RMSE is '+str(test_rmse_basiclr_std))
lr.fit(data_std, target)
pred_basiclr_std = lr.predict(data_std)
plot2(target, pred_basiclr_std, 'using basic linear regression with standardizing')
print ('-'*30)

# iii
F, pval = fs.f_regression(data, target)
print ('Using f_regression, the F values are: ')
print (F)
print ('Using f_regression, the p values are: ')
print (pval)
data_freg = data[:, 1:4]
data_freg_folds = kfold(data_freg)
train_rmse_basiclr_freg, test_rmse_basiclr_freg = cv(data_freg_folds, target_folds, lr)
print ('For basic linear regression model after feature selection using f_regression, the average training RMSE is '+str(train_rmse_basiclr_freg)+', average test RMSE is '+str(test_rmse_basiclr_freg))
lr.fit(data_freg, target)
pred_basiclr_freg = lr.predict(data_freg)
plot2(target, pred_basiclr_freg, 'using basic linear regression after feature selection using f_regression')
print ('-'*20)

mi = fs.mutual_info_regression(data, target)
print ('Using mutual information regression, the estimated mutual information between each feature and the target are: ')
print (mi)
data_mif = data[:, 2:5]
data_mif_folds = kfold(data_mif)
train_rmse_basiclr_mif, test_rmse_basiclr_mif = cv(data_mif_folds, target_folds, lr)
print ('For basic linear regression model after feature selection using mutual information regression, the average training RMSE is '+str(train_rmse_basiclr_mif)+', average test RMSE is '+str(test_rmse_basiclr_mif))
lr.fit(data_mif, target)
pred_basiclr_mif = lr.predict(data_mif)
plot2(target, pred_basiclr_mif, 'using basic linear regression after feature selection using mutual information regression')
print ('-'*30)

# iv
masks = []
for i in range(32):
    masks.append((list(bin(i)[2:].zfill(5))))
for mask in masks:
    for i in range(5):
        if mask[i] == '1':
            mask[i] = True
        else:
            mask[i] = False

train_enc = []
test_enc = []
print ('Doing linear regression in 32 combinations...')
for i in range(32):
    enc = pre.OneHotEncoder(categorical_features = masks[i])
    data_enc = enc.fit_transform(data)
    if True in masks[i]:
        data_enc = data_enc.toarray().astype(int)
    else:
        data_enc = data_enc.astype(int)
    data_enc_folds = kfold(data_enc)
    train_rmse_enc, test_rmse_enc = cv(data_enc_folds, target_folds, lr)
    train_enc.append(train_rmse_enc)
    test_enc.append(test_rmse_enc)
    print ('Combination '+str(i+1)+' is done!')
combination = list(range(1, 33))
plt.plot(combination, train_enc, combination, test_enc)
plt.xlabel('combination')
plt.ylabel('average RMSE')
plt.title('combination vs average RMSE using shuffled data')
plt.legend(['training RMSE', 'test RMSE'])
plt.show()
print ('-'*20)

data_noshuffle = np.asarray(scalar)[:, 0:5].astype(int)
target_noshuffle = np.asarray(scalar)[:, 5].reshape(1, data_noshuffle.shape[0]).tolist()[0]
_, target_noshuffle_folds = kfold(data_noshuffle, target_noshuffle)
train_enc_noshuffle = []
test_enc_noshuffle = []

print ('Doing linear regression in 32 combinations...')
lowest_test_rmse_lr = 1
for i in range(32):
    enc = pre.OneHotEncoder(categorical_features = masks[i])
    data_enc_noshuffle = enc.fit_transform(data_noshuffle)
    if True in masks[i]:
        data_enc_noshuffle = data_enc_noshuffle.toarray().astype(int)
    else:
        data_enc_noshuffle = data_enc_noshuffle.astype(int)
    data_enc_noshuffle_folds = kfold(data_enc_noshuffle)
    train_rmse_enc_noshuffle, test_rmse_enc_noshuffle = cv(data_enc_noshuffle_folds, target_noshuffle_folds, lr)
    train_enc_noshuffle.append(train_rmse_enc_noshuffle)
    test_enc_noshuffle.append(test_rmse_enc_noshuffle)
    if test_rmse_enc_noshuffle < lowest_test_rmse_lr:
        lowest_test_rmse_lr = test_rmse_enc_noshuffle
        bestmask_lr = masks[i]
    print ('Combination '+str(i+1)+' is done!')
print ('Combination is: '+str(bestmask_lr))
print ('Lowest test RMSE in basic linear regression is ' + str(lowest_test_rmse_lr))
plt.plot(combination, train_enc_noshuffle, combination, test_enc_noshuffle)
plt.xlabel('combination')
plt.ylabel('average RMSE')
plt.title('combination vs average RMSE using non-shuffled data')
plt.legend(['training RMSE', 'test RMSE'])
plt.show()
print ('-'*30)

# v
print ('Fitted coeffcients of lagre increases in test RMSE are: ')
large_diff_masks = [masks[21], masks[23], masks[29], masks[31]]
for mask in large_diff_masks:
    enc = pre.OneHotEncoder(categorical_features = mask)
    data_enc_large = enc.fit_transform(data_noshuffle)
    lr.fit(data_enc_large, target_noshuffle)
    print ('Combination is: '+str(mask))
    print (lr.coef_, lr.intercept_)
print ('-'*20)
   
alpha_list = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
lowest_test_rmse_ridge = 1
for alpha in alpha_list:
    ridge = linear_model.Ridge(alpha = alpha)
    for i in range(32):
        enc = pre.OneHotEncoder(categorical_features = masks[i])
        data_enc_noshuffle = enc.fit_transform(data_noshuffle)
        if True in masks[i]:
            data_enc_noshuffle = data_enc_noshuffle.toarray().astype(int)
        else:
            data_enc_noshuffle = data_enc_noshuffle.astype(int)
        data_enc_noshuffle_folds = kfold(data_enc_noshuffle)
        _, test_rmse_ridge = cv(data_enc_noshuffle_folds, target_noshuffle_folds, ridge)
        if test_rmse_ridge < lowest_test_rmse_ridge:
            lowest_test_rmse_ridge = test_rmse_ridge
            bestmask_ridge = masks[i]
            bestalpha_ridge = alpha
    print ('Alpha = '+str(alpha)+' is done!')
print ('Combination is: '+str(bestmask_ridge))
print ('Lowest test RMSE with Ridge Regularizer is '+str(lowest_test_rmse_ridge)+', alpha = '+str(bestalpha_ridge))
print ('-'*20)

alpha_list = [1e-4, 1e-3, 1e-2]
lowest_test_rmse_lasso = 1
for alpha in alpha_list:
    lasso = linear_model.Lasso(alpha = alpha)
    for i in range(32):
        enc = pre.OneHotEncoder(categorical_features = masks[i])
        data_enc_noshuffle = enc.fit_transform(data_noshuffle)
        if True in masks[i]:
            data_enc_noshuffle = data_enc_noshuffle.toarray().astype(int)
        else:
            data_enc_noshuffle = data_enc_noshuffle.astype(int)
        data_enc_noshuffle_folds = kfold(data_enc_noshuffle)
        _, test_rmse_lasso = cv(data_enc_noshuffle_folds, target_noshuffle_folds, lasso)
        if test_rmse_lasso < lowest_test_rmse_lasso:
            lowest_test_rmse_lasso = test_rmse_lasso
            bestmask_lasso = masks[i]
            bestalpha_lasso = alpha
    print ('Alpha = '+str(alpha)+' is done!')
print ('Combination is: '+str(bestmask_lasso))
print ('Lowest test RMSE with Lasso Regularizer is '+str(lowest_test_rmse_lasso)+', alpha = '+str(bestalpha_lasso))
print ('-'*20)

lambda1_list = [1e-4, 1e-3, 1e-2]
lambda2_list = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
lowest_test_rmse_elasticnet = 1
for lambda1 in lambda1_list:
    for lambda2 in lambda2_list:
        elasticnet = linear_model.ElasticNet(alpha = (lambda1+lambda2), l1_ratio = lambda1/(lambda1+lambda2))
        for i in range(32):
            enc = pre.OneHotEncoder(categorical_features = masks[i])
            data_enc_noshuffle = enc.fit_transform(data_noshuffle)
            if True in masks[i]:
                data_enc_noshuffle = data_enc_noshuffle.toarray().astype(int)
            else:
                data_enc_noshuffle = data_enc_noshuffle.astype(int)
            data_enc_noshuffle_folds = kfold(data_enc_noshuffle)
            _, test_rmse_elasticnet = cv(data_enc_noshuffle_folds, target_noshuffle_folds, elasticnet)
            if test_rmse_elasticnet < lowest_test_rmse_elasticnet:
                lowest_test_rmse_elasticnet = test_rmse_elasticnet
                bestmask_elasticnet = masks[i]
                bestlambda1 = lambda1
                bestlambda2 = lambda2
        print ('lambda1 = '+str(lambda1)+', lambda2 = '+str(lambda2)+' is done!')
print ('Combination is: '+str(bestmask_elasticnet))
print ('Lowest test RMSE with Elastic Net Regularizer is '+str(lowest_test_rmse_elasticnet)+', lambda1 = '+str(bestlambda1)+', lambda2 = '+str(bestlambda2))
print ('-'*20)

enc = pre.OneHotEncoder(categorical_features = bestmask_lr)
data_bestlr = enc.fit_transform(data_noshuffle)
lr.fit(data_bestlr, target_noshuffle)
pred_bestlr = lr.predict(data_bestlr)
print ('Using combination'+str(bestmask_lr)+', fitted coefficients of lowest test RMSE using basic linear regression are: ')
print (lr.coef_, lr.intercept_)
plot2(target_noshuffle, pred_bestlr, 'using basic linear regression in 32 combinations')
print ('-'*20)

ridge = linear_model.Ridge(alpha = bestalpha_ridge)
enc = pre.OneHotEncoder(categorical_features = bestmask_ridge)
data_ridge = enc.fit_transform(data_noshuffle)
ridge.fit(data_ridge, target_noshuffle)
pred_ridge = ridge.predict(data_ridge)
print ('Using combination'+str(bestmask_ridge)+', fitted coefficients of lowest test RMSE with Ridge Regularizer are: ')
print (ridge.coef_, ridge.intercept_)
plot2(target_noshuffle, pred_ridge, 'using Ridge Regularizer in 32 combinations')
print ('-'*20)

lasso = linear_model.Lasso(alpha = bestalpha_lasso)
enc = pre.OneHotEncoder(categorical_features = bestmask_lasso)
data_lasso = enc.fit_transform(data_noshuffle)
lasso.fit(data_lasso, target_noshuffle)
pred_lasso = lasso.predict(data_lasso)
print ('Using combination'+str(bestmask_lasso)+', fitted coefficients of lowest test RMSE with Lasso Regularizer are: ')
print (lasso.coef_, lasso.intercept_)
plot2(target_noshuffle, pred_lasso, 'using Lasso Regularizer in 32 combinations')
print ('-'*20)

elasticnet = linear_model.ElasticNet(alpha = (bestlambda1+bestlambda2), l1_ratio = bestlambda1/(bestlambda1+bestlambda2))
enc = pre.OneHotEncoder(categorical_features = bestmask_elasticnet)
data_elasticnet = enc.fit_transform(data_noshuffle)
elasticnet.fit(data_elasticnet, target_noshuffle)
pred_elasticnet = elasticnet.predict(data_elasticnet)
print ('Using combination'+str(bestmask_elasticnet)+', fitted coefficients of lowest test RMSE with Elastic Net Regularizer are: ')
print (elasticnet.coef_, elasticnet.intercept_)
plot2(target_noshuffle, pred_elasticnet, 'using Elastic Net Regularizer in 32 combinations')
print ('-'*30)

# 2d i
workflows = []
for i in range(5):
    workflows.append([])
for i in range(len(scalar)):
    workflows[scalar[i][3]].append(scalar[i])
for i in range(5):
    lr = linear_model.LinearRegression()
    tmp = np.asarray(workflows[i])
    np.random.seed(1)
    np.random.shuffle(tmp)
    data_flow = tmp[:, 0:5].astype(int)
    target_flow  = tmp[:, 5].reshape(1, data_flow.shape[0]).tolist()[0]
    data_flow_folds, target_flow_folds = kfold(data_flow, target_flow)
    train_rmse_flow, test_rmse_flow = cv(data_flow_folds, target_flow_folds, lr)
    print ('For workflow '+str(i)+', using basic linear regression, the average training RMSE is '+str(train_rmse_flow)+', average test RMSE is '+str(test_rmse_flow))
    lr.fit(data_flow, target_flow)
    pred_flow = lr.predict(data_flow)
    plot2(target_flow, pred_flow, 'in workflow '+str(i))
