import numpy as np
import pdb
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
import scipy.sparse.linalg as ssl
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler

categories = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
twenty_all = fetch_20newsgroups(subset = 'all', categories = categories, shuffle = True, random_state = 42)

# question 1
min_df = 3
stop_words = text.ENGLISH_STOP_WORDS

punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

def five_measure_scores (label_true, label_pred):
    print ("Homogeneity_score = %f" % homogeneity_score(label_true, label_pred))
    print ("Completeness_score = %f" % completeness_score(label_true, label_pred))
    print ("Adjusted_rand_score = %f" % adjusted_rand_score(label_true, label_pred))
    print ("V_measure_score = %f" % v_measure_score(label_true, label_pred))
    print ("Adjusted_mutual_info_score = %f" % adjusted_mutual_info_score(label_true, label_pred))

for i in range(len(twenty_all.data)):
    twenty_all.data[i] = ''.join([s for s in twenty_all.data[i] if s.isalpha() or s.isspace() or s in punctuation])
    for ch in punctuation:
        twenty_all.data[i] = twenty_all.data[i].replace(ch, ' ')

vectorizer_count = CountVectorizer(min_df = min_df, stop_words = stop_words)
all_count = vectorizer_count.fit_transform(twenty_all.data)

tfidf_transformer = TfidfTransformer()
all_tfidf = tfidf_transformer.fit_transform(all_count)

print ("The dimensions of the TF-IDF matrix:")
print (all_tfidf.shape)
print ('-'*30)

# question 2
kmeans = KMeans(n_clusters = 2, random_state = 42)
pred_tfidf = kmeans.fit_predict(all_tfidf)

# put 8 subclasses into 2 classes
label_all = []
two_classes = ['comp', 'rec']

for i in range(len(twenty_all.data)):
    if 'comp' in twenty_all.filenames[i]:
        label_all.append(0)
    else:
        label_all.append(1)

print ("Contingency_matrix: ")
print (contingency_matrix(label_all, pred_tfidf))
five_measure_scores(label_all, pred_tfidf)
print ('-'*30)

# question 3
r = 1000
total = np.trace(np.dot(all_tfidf.toarray(), all_tfidf.toarray().T))

u, s, vt = ssl.svds(all_tfidf, k = r)
s[:] = s[::-1]
u[:, :] = u[:, ::-1]
vt[:, :] = vt[::-1, :]
eigval = np.square(s)
percent = []
for i in range(1, r+1):
    tmp = sum(eigval[:i]) / total
    percent.append(tmp)
x = range(1, r+1)

plt.plot(x, percent)
plt.xlabel('r')
plt.ylabel('percent')
plt.title('percent of variance of top r principle components can retain')
plt.show()

rs = [1, 2, 3, 5, 10, 20, 50, 100, 300]
#rs = list(range(1, 11))
homogeneity = []
completeness = []
adjusted_rand = []
v_measure =[]
adjusted_mutual_info = []

# for LSI
for r in rs:
    tic = time.time()
    SVD = TruncatedSVD(n_components = r, algorithm = 'arpack')
    all_LSI = SVD.fit_transform(all_tfidf)
    pred = kmeans.fit_predict(all_LSI)
    print ("Contingency matrix when r = " + str(r) + " using LSI")
    print (contingency_matrix(label_all, pred))
    homogeneity.append(homogeneity_score(label_all, pred))
    completeness.append(completeness_score(label_all, pred))
    adjusted_rand.append(adjusted_rand_score(label_all, pred))
    v_measure.append(v_measure_score(label_all, pred))
    adjusted_mutual_info.append(adjusted_mutual_info_score(label_all, pred))
    toc = time.time()
    print ("It took " + str(toc - tic) + "s to calculate.")
    print ('-'*20)
print ('-'*30)
plt.plot(rs, homogeneity)
plt.plot(rs, completeness)
plt.plot(rs, adjusted_rand)
plt.plot(rs, v_measure)
plt.plot(rs, adjusted_mutual_info)
plt.xlabel('r')
plt.ylabel('clustering purity metrics')
plt.title('r vs clustering purity using LSI')
plt.legend(['homogeneity', 'completeness', 'adjusted_rand', 'v_measure', 'adjusted_mutual_info'])
plt.show()

homogeneity = []
completeness = []
adjusted_rand = []
v_measure =[]
adjusted_mutual_info = []

# for NMF
for r in rs:
    tic = time.time()
    nmf = NMF(n_components = r, init = 'random', random_state = 0)
    all_NMF = nmf.fit_transform(all_tfidf)
    pred = kmeans.fit_predict(all_NMF)
    print ("Contingency matrix when r = " + str(r) + " using NMF")
    print (contingency_matrix(label_all, pred))
    homogeneity.append(homogeneity_score(label_all, pred))
    completeness.append(completeness_score(label_all, pred))
    adjusted_rand.append(adjusted_rand_score(label_all, pred))
    v_measure.append(v_measure_score(label_all, pred))
    adjusted_mutual_info.append(adjusted_mutual_info_score(label_all, pred))
    toc = time.time()
    print ("It took " + str(toc - tic) + "s to calculate.")
    print ('-'*20)
print ('-'*30)
plt.plot(rs, homogeneity)
plt.plot(rs, completeness)
plt.plot(rs, adjusted_rand)
plt.plot(rs, v_measure)
plt.plot(rs, adjusted_mutual_info)
plt.xlabel('r')
plt.ylabel('clustering purity metrics')
plt.title('r vs clustering purity using NMF')
plt.legend(['homogeneity', 'completeness', 'adjusted_rand', 'v_measure', 'adjusted_mutual_info'])
plt.show()

#question 4
SVD = TruncatedSVD(n_components = 4, algorithm = 'arpack')
best_r_LSI = SVD.fit_transform(all_tfidf)
pred_km_LSI = kmeans.fit_predict(best_r_LSI)
centers_LSI = kmeans.cluster_centers_
nmf = NMF(n_components = 3, init = 'random', random_state = 0)
best_r_NMF = nmf.fit_transform(all_tfidf)
pred_km_NMF = kmeans.fit_predict(best_r_NMF)
centers_NMF = kmeans.cluster_centers_

def visualize_clustering_results(label_true, label_pred, X_2d, centers, method):
    color = ["r", "g"]
    mark = ["o", "+"]

    for i in range(len(label_all)):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], s=12, marker=mark[label_true[i]], color=color[label_pred[i]], alpha=0.5)
    for i in range(2):
        plt.scatter(centers[i, 0], centers[i, 1], marker='x', s=100, linewidths=5, color='k', alpha=0.6)
    plt.title('Clustering results of ' + method)
    plt.show()

visualize_clustering_results(label_all, pred_km_LSI, best_r_LSI, centers_LSI, 'LSI with best r = 4')
visualize_clustering_results(label_all, pred_km_NMF, best_r_NMF, centers_NMF, 'NMF with best r = 3')

#normalize
scaler = StandardScaler(copy = False, with_mean = False)
best_r_LSI_norm = scaler.fit_transform(best_r_LSI)
pred_km_LSI_norm = kmeans.fit_predict(best_r_LSI_norm)
centers_LSI_norm = kmeans.cluster_centers_
print ('LSI with normalizer: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label_all, pred_km_LSI_norm))
five_measure_scores(label_all, pred_km_LSI_norm)
print ('-'*30)
visualize_clustering_results(label_all, pred_km_LSI_norm, best_r_LSI_norm, centers_LSI_norm, 'LSI with normalizer (r = 4)')

best_r_NMF_norm = scaler.fit_transform(best_r_NMF)
pred_km_NMF_norm = kmeans.fit_predict(best_r_NMF_norm)
centers_NMF_norm = kmeans.cluster_centers_
print ('NMF with normalizer: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label_all, pred_km_NMF_norm))
five_measure_scores(label_all, pred_km_NMF_norm)
print ('-'*30)
visualize_clustering_results(label_all, pred_km_NMF_norm, best_r_NMF_norm, centers_NMF_norm, 'NMF with normalizer (r = 3)')

#logarithm transformation for only NMF
best_r_NMF_log = np.log(best_r_NMF + 0.01)
pred_km_NMF_log = kmeans.fit_predict(best_r_NMF_log)
centers_NMF_log = kmeans.cluster_centers_
print ('NMF with logarithm: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label_all, pred_km_NMF_log))
five_measure_scores(label_all, pred_km_NMF_log)
print ('-'*30)
visualize_clustering_results(label_all, pred_km_NMF_log, best_r_NMF_log, centers_NMF_log, 'NMF with logarithm (r = 3)')

#normalize + logarithm transformation
best_r_NMF_norm_log = np.log(best_r_NMF_norm + 0.01)
pred_km_NMF_norm_log = kmeans.fit_predict(best_r_NMF_norm_log)
centers_NMF_norm_log = kmeans.cluster_centers_
print ('NMF with normalizer + logarithm: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label_all, pred_km_NMF_norm_log))
five_measure_scores(label_all, pred_km_NMF_norm_log)
print ('-'*30)
visualize_clustering_results(label_all, pred_km_NMF_norm_log, best_r_NMF_norm_log, centers_NMF_norm_log, 'NMF with normalizer + logarithm (r = 3)')

#logarithm transformation + normalize
best_r_NMF_log_norm = scaler.fit_transform(best_r_NMF_log)
pred_km_NMF_log_norm = kmeans.fit_predict(best_r_NMF_log_norm)
centers_NMF_log_norm = kmeans.cluster_centers_
print ('NMF with logarithm + normalizer: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label_all, pred_km_NMF_log_norm))
five_measure_scores(label_all, pred_km_NMF_log_norm)
print ('-'*30)
visualize_clustering_results(label_all, pred_km_NMF_log_norm, best_r_NMF_log_norm, centers_NMF_log_norm, 'NMF with logarithm + normalizer (r = 3)')

# question 5
classes_20 = ['alt.atheism',
              'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
              'misc.forsale',              
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
              'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian',
              'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc']
all20 = fetch_20newsgroups(subset = 'all', categories = classes_20, shuffle = True, random_state = 42)
label = all20.target

for i in range(len(all20.data)):
    all20.data[i] = ''.join([s for s in all20.data[i] if s.isalpha() or s.isspace() or s in punctuation])
    for ch in punctuation:
        all20.data[i] = all20.data[i].replace(ch, ' ')
        
vectorizer_count = CountVectorizer(min_df = min_df, stop_words = stop_words)
all20_count = vectorizer_count.fit_transform(all20.data)

tfidf_transformer = TfidfTransformer()
all20_tfidf = tfidf_transformer.fit_transform(all20_count)

print ("The dimensions of the TF-IDF matrix:")
print (all20_tfidf.shape)
print ('-'*30)

kmeans = KMeans(n_clusters = 20, random_state = 42)

# Find the best r for LSI and NMF
rs = list(range(1, 16))
homogeneity = []
completeness = []
adjusted_rand = []
v_measure =[]
adjusted_mutual_info = []

for r in rs:
    SVD = TruncatedSVD(n_components = r, algorithm = 'arpack')
    all20_LSI = SVD.fit_transform(all20_tfidf)
    pred = kmeans.fit_predict(all20_LSI)
    homogeneity.append(homogeneity_score(label, pred))
    completeness.append(completeness_score(label, pred))
    adjusted_rand.append(adjusted_rand_score(label, pred))
    v_measure.append(v_measure_score(label, pred))
    adjusted_mutual_info.append(adjusted_mutual_info_score(label, pred))
plt.plot(rs, homogeneity)
plt.plot(rs, completeness)
plt.plot(rs, adjusted_rand)
plt.plot(rs, v_measure)
plt.plot(rs, adjusted_mutual_info)
plt.xlabel('r')
plt.ylabel('clustering purity metrics')
plt.title('r vs clustering purity using LSI for 20 classes')
plt.legend(['homogeneity', 'completeness', 'adjusted_rand', 'v_measure', 'adjusted_mutual_info'])
plt.show()

homogeneity = []
completeness = []
adjusted_rand = []
v_measure =[]
adjusted_mutual_info = []

for r in rs:
    nmf = NMF(n_components = r, init = 'random', random_state = 0)
    all20_NMF = nmf.fit_transform(all20_tfidf)
    pred = kmeans.fit_predict(all20_NMF)
    homogeneity.append(homogeneity_score(label, pred))
    completeness.append(completeness_score(label, pred))
    adjusted_rand.append(adjusted_rand_score(label, pred))
    v_measure.append(v_measure_score(label, pred))
    adjusted_mutual_info.append(adjusted_mutual_info_score(label, pred))
plt.plot(rs, homogeneity)
plt.plot(rs, completeness)
plt.plot(rs, adjusted_rand)
plt.plot(rs, v_measure)
plt.plot(rs, adjusted_mutual_info)
plt.xlabel('r')
plt.ylabel('clustering purity metrics')
plt.title('r vs clustering purity using NMF for 20 classes')
plt.legend(['homogeneity', 'completeness', 'adjusted_rand', 'v_measure', 'adjusted_mutual_info'])
plt.show()

SVD = TruncatedSVD(n_components = 9, algorithm = 'arpack')
best_r_LSI = SVD.fit_transform(all20_tfidf)
pred_km_LSI = kmeans.fit_predict(best_r_LSI)
centers_LSI = kmeans.cluster_centers_
nmf = NMF(n_components = 8, init = 'random', random_state = 0)
best_r_NMF = nmf.fit_transform(all20_tfidf)
pred_km_NMF = kmeans.fit_predict(best_r_NMF)
centers_NMF = kmeans.cluster_centers_
print ('Optimizing...')

def visualize_clustering_results(label_true, label_pred, X_2d, centers, method):
    color = ["grey", "lightcoral", "maroon", "mistyrose", "coral", "peachpuff", "darkorange", "orange", "darkgoldenrod",
             "olive", "yellowgreen", "lawngreen", "lightgreen", "g", "mediumseagreen", "mediumturquoise", "c", "cadetblue",
             "skyblue", "dodgerblue"]
    mark = ["x", "D", "d", ".", ",", "o", "v", "^", "<", ">" , "1", "2", "3", "4", "s", "p", "*", "h", "H", "+"]

    for i in range(len(label_true)):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], s=12, marker=mark[label_true[i]], color=color[label_pred[i]], alpha=0.5)
    for i in range(2):
        plt.scatter(centers[i, 0], centers[i, 1], marker='x', s=100, linewidths=5, color='k', alpha=0.6)
    plt.title('Clustering results of ' + method)
    plt.show()

visualize_clustering_results(label, pred_km_LSI, best_r_LSI, centers_LSI, 'LSI with best r = 9')
visualize_clustering_results(label, pred_km_NMF, best_r_NMF, centers_NMF, 'NMF with best r = 8')

#normalize
scaler = StandardScaler(copy = False, with_mean = False)
best_r_LSI_norm = scaler.fit_transform(best_r_LSI)
pred_km_LSI_norm = kmeans.fit_predict(best_r_LSI_norm)
centers_LSI_norm = kmeans.cluster_centers_
print ('LSI with normalizer: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label, pred_km_LSI_norm))
five_measure_scores(label, pred_km_LSI_norm)
print ('-'*30)
visualize_clustering_results(label, pred_km_LSI_norm, best_r_LSI_norm, centers_LSI_norm, 'LSI with normalizer (r = 9)')

best_r_NMF_norm = scaler.fit_transform(best_r_NMF)
pred_km_NMF_norm = kmeans.fit_predict(best_r_NMF_norm)
centers_NMF_norm = kmeans.cluster_centers_
print ('NMF with normalizer: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label, pred_km_NMF_norm))
five_measure_scores(label, pred_km_NMF_norm)
print ('-'*30)
visualize_clustering_results(label, pred_km_NMF_norm, best_r_NMF_norm, centers_NMF_norm, 'NMF with normalizer (r = 8)')

#logarithm transformation for only NMF
best_r_NMF_log = np.log(best_r_NMF + 0.01)
pred_km_NMF_log = kmeans.fit_predict(best_r_NMF_log)
centers_NMF_log = kmeans.cluster_centers_
print ('NMF with logarithm: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label, pred_km_NMF_log))
five_measure_scores(label, pred_km_NMF_log)
print ('-'*30)
visualize_clustering_results(label, pred_km_NMF_log, best_r_NMF_log, centers_NMF_log, 'NMF with logarithm (r = 8)')

#normalize + logarithm transformation
best_r_NMF_norm_log = np.log(best_r_NMF_norm + 0.01)
pred_km_NMF_norm_log = kmeans.fit_predict(best_r_NMF_norm_log)
centers_NMF_norm_log = kmeans.cluster_centers_
print ('NMF with normalizer + logarithm: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label, pred_km_NMF_norm_log))
five_measure_scores(label, pred_km_NMF_norm_log)
print ('-'*30)
visualize_clustering_results(label, pred_km_NMF_norm_log, best_r_NMF_norm_log, centers_NMF_norm_log, 'NMF with normalizer + logarithm (r = 8)')

#logarithm transformation + normalize
best_r_NMF_log_norm = scaler.fit_transform(best_r_NMF_log)
pred_km_NMF_log_norm = kmeans.fit_predict(best_r_NMF_log_norm)
centers_NMF_log_norm = kmeans.cluster_centers_
print ('NMF with logarithm + normalizer: ')
print ("Contingency_matrix: ")
print (contingency_matrix(label, pred_km_NMF_log_norm))
five_measure_scores(label, pred_km_NMF_log_norm)
print ('-'*30)
visualize_clustering_results(label, pred_km_NMF_log_norm, best_r_NMF_log_norm, centers_NMF_log_norm, 'NMF with logarithm + normalizer (r = 8)')