import csv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sklearn.metrics as metrics

user_num = 671
movie_num = 9125
R = np.zeros([user_num, movie_num])
movieid = []

with open('movies.csv') as csvfile:
    tmp = csv.reader(csvfile, delimiter=',')
    for row in tmp:
        if row[0] != 'movieId':
            movieid.append(int(row[0]))
   
with open('ratings.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] != 'userId':
            R[int(row[0])-1, movieid.index(int(row[1]))] = float(row[2])

# question 1
available_ratings = np.count_nonzero(R)
possible_ratings = user_num * movie_num
sparsity = available_ratings / possible_ratings
print("Sparsity is " + str(sparsity))

# question 2
rating_values = np.arange(0.5, 6, 0.5)
freq_rating_values = []
with open('ratings.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] != 'userId':
            freq_rating_values.append(float(row[2]))
plt.hist(freq_rating_values, bins = rating_values)
plt.xticks(rating_values)
plt.xlabel('ratings')
plt.ylabel('frequency')
plt.title('frequency of the rating values')
plt.grid(True)
plt.show()

# question 3
ratings_received = np.count_nonzero(R, axis = 0)
tmp = ratings_received.argsort()[::-1]
id = []
times = []
for i in range(movie_num):
    id.append(str(movieid[tmp[i]]))
    times.append(ratings_received[tmp[i]])
x = range(len(times))
plt.plot(x, times, 'ro-')
plt.xticks([])
plt.ylabel('number of ratings received')
plt.title('number of ratings a movie received')
plt.show()

# question 4
ratings_gave = np.count_nonzero(R, axis = 1)
tmp = ratings_gave.argsort()[::-1]
id = []
times = []
for i in range(user_num):
    id.append(str(tmp[i]+1))
    times.append(ratings_gave[tmp[i]])
x = range(len(times))
plt.plot(x, times, 'ro-')
plt.xticks([])
plt.ylabel('number of ratings gave')
plt.title('number of ratings a user gave')
plt.show()

# question 6
var = np.var(R, axis = 0)
plt.hist(var, bins = np.arange(0, 6, 0.5))
plt.xticks(np.arange(0, 6, 0.5))
plt.xlabel('rating variance')
plt.ylabel('number of movies')
plt.title('number of movies with certain rating variance')
plt.grid(True)
plt.show()

# question 30
def naive(data, test):
    mean = np.mean(data)
    if isinstance(test, list):
        pred = []
        for i in range(len(test)):
            pred.append(mean)
        return pred
    else:
        return mean

def extract(a):
    tmp = []
    for i in range(len(a)):
        if a[i] != 0:
            tmp.append(a[i])
    return tmp

def naive_filter(R):
    rmse_naive = 0
    num_folds = 10
    size = R.shape[0]
    for i in range(size):
        extracted = extract(R[i])
        if len(extracted) == 0 :
            size -= 1
            continue
        elif len(extracted) == 1:
            continue
        else:
            folds = []
            tmp = 0
            if len(extracted) < 10:
                cv = len(extracted)
                for j in range(cv):
                    folds.append(extracted[j])
                for k in range(cv):
                    test = folds[k]
                    data = []
                    for m in range(cv):
                        if m != k:
                            data.append(folds[m])
                    pred = naive(data, test)
                    tmp += abs(test-pred)
            else:
                cv = num_folds
                for j in range(cv):
                    folds.append(extracted[int(j*len(extracted)/cv):int((j+1)*len(extracted)/num_folds)])
                for k in range(cv):
                    test = folds[k]
                    data = []
                    for m in range(cv):
                        if m != k:
                            data.extend(folds[m])
                    pred = naive(data, test)
                    tmp += np.sqrt(metrics.mean_squared_error(test, pred))
            rmse_naive += tmp/cv
    return rmse_naive/size
    
rmse_naive = naive_filter(R)
print ('Average RMSE using Naive collaborative filtering prediction is ' + str(rmse_naive))

# question 31
def naive_filter_popular(R):
    tmp = R.copy()
    for i in range(tmp.shape[1]):
        if np.count_nonzero(tmp[:, i]) < 3:
            tmp[:, i] = 0
    return naive_filter(tmp)

rmse_naive_popular = naive_filter_popular(R)
print ('Average RMSE using Naive collaborative filtering prediction in popular set is ' + str(rmse_naive_popular))

# question 32
def naive_filter_unpopular(R):
    tmp = R.copy()
    for i in range(tmp.shape[1]):
        if np.count_nonzero(tmp[:, i]) > 2:
            tmp[:, i] = 0
    return naive_filter(tmp)

rmse_naive_unpopular = naive_filter_unpopular(R)
print ('Average RMSE using Naive collaborative filtering prediction in unpopular set is ' + str(rmse_naive_unpopular))

# question 32
def naive_filter_highvar(R):
    tmp = R.copy()
    for i in range(tmp.shape[1]):
        if np.count_nonzero(tmp[:, i]) < 5:
            tmp[:, i] = 0
        if var[i] < 2:
            tmp[:, i] = 0
    return naive_filter(tmp)
    
rmse_naive_highvar = naive_filter_highvar(R)
print ('Average RMSE using Naive collaborative filtering prediction in high variance set is ' + str(rmse_naive_highvar))