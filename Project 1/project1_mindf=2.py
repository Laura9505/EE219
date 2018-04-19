import numpy as np
import matplotlib.pyplot as plt
#a
from sklearn.datasets import fetch_20newsgroups
#b
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
from sklearn.feature_extraction.text import TfidfTransformer
#d
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
#e
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc, classification_report
#g
from sklearn.naive_bayes import MultinomialNB
#h
from sklearn import linear_model
#i
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# question a)
categories = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)
twenty_test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42)
twenty_all = fetch_20newsgroups(subset = 'all', categories = categories, shuffle = True, random_state = 42)

fig, ax = plt.subplots()
width = 0.8
num = np.arange(8)
    
plt.hist(twenty_train.target, 15, alpha = 0.75, color = 'green', width = 0.8)
ax.set_xlim(-0.5, 9)
ax.set_ylim(0, 800)
ax.set_ylabel('Document Number')
ax.set_title('The Number of Training Documents Per Class')
xtickmarks = categories
ax.set_xticks(num)
xticknames = ax.set_xticklabels(xtickmarks)
plt.setp(xticknames, rotation = 90)
#plt.show()

# question b)
min_df = 2
print ('The following results are from min_df = ' + str(min_df))
print ('-'*30)
print ('extracting data...')

stop_words = text.ENGLISH_STOP_WORDS

# exclude punctuations
punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

for i in range(len(twenty_train.data)):
    twenty_train.data[i] = ''.join([s for s in twenty_train.data[i] if s.isalpha() or s.isspace() or s in punctuation])
    for ch in punctuation:
        twenty_train.data[i] = twenty_train.data[i].replace(ch, ' ')
for i in range(len(twenty_test.data)):
    twenty_test.data[i] = ''.join([s for s in twenty_test.data[i] if s.isalpha() or s.isspace() or s in punctuation])
    for ch in punctuation:
        twenty_test.data[i] = twenty_test.data[i].replace(ch, ' ')

#using stemmed version of words
sbs = nltk.stem.SnowballStemmer('english')

for i in range(len(twenty_train.data)):
    words = twenty_train.data[i].split()
    for j in range(len(words)):
        words[j] = sbs.stem(words[j])
    twenty_train.data[i] = str(words)
for i in range(len(twenty_test.data)):
    words = twenty_test.data[i].split()
    for j in range(len(words)):
        words[j] = sbs.stem(words[j])
    twenty_test.data[i] = str(words)    
  
vectorizer_count = CountVectorizer(min_df = min_df, stop_words = stop_words)
train_count = vectorizer_count.fit_transform(twenty_train.data)
test_count = vectorizer_count.transform(twenty_test.data)
print ("extracted terms: %d" %(len(vectorizer_count.get_feature_names())))
print ('-'*30)

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_count)
test_tfidf = tfidf_transformer.transform(test_count)
'''
#question c)
print ('extracting 10 most significant terms...')
classes_4 = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
taridx = [3, 4, 6, 15]
classes_20 = ['alt.atheism',
              'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
              'misc.forsale',              
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
              'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian',
              'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc']
twenty_train_c = fetch_20newsgroups(subset='train', categories=classes_20, shuffle = True, random_state = 42)

for i in range(len(twenty_train_c.data)):
    twenty_train_c.data[i] = ''.join([s for s in twenty_train_c.data[i] if s.isalpha() or s.isspace() or s in punctuation])
    for ch in punctuation:
        twenty_train_c.data[i] = twenty_train_c.data[i].replace(ch, ' ')

# using stemmed version of words
sbs = nltk.stem.SnowballStemmer('english')

for i in range(len(twenty_train_c.data)):
    words = twenty_train_c.data[i].split()
    for j in range(len(words)):
        words[j] = sbs.stem(words[j])
    twenty_train_c.data[i] = (words)

#putting documents from the same category in one document
firstappear = []
for i in range(20):
    firstappear.append(list(twenty_train_c.target).index(i))
firstappear_sorted = sorted(firstappear)
four_classes = []
for i in range(4):
    four_classes.append(firstappear_sorted.index(firstappear[taridx[i]]))

for i in range(20):
    for j in range(firstappear[i]+1, len(twenty_train_c.data)):
        if twenty_train_c.target[j] == i:
            twenty_train_c.data[firstappear[i]].extend(twenty_train_c.data[j])
            twenty_train_c.data[j] = []
while [] in twenty_train_c.data:
    twenty_train_c.data.remove([])
for i in range(len(twenty_train_c.data)):
    twenty_train_c.data[i] = str(twenty_train_c.data[i])

#counting
vectorizer_c = CountVectorizer(min_df=5, stop_words=stop_words)
train_count_c = vectorizer_c.fit_transform(twenty_train_c.data)
tficf_transformer = TfidfTransformer()
train_tficf = tficf_transformer.fit_transform(train_count_c)
tficf = train_tficf.toarray()

idx10 = []
for i in range(4):
    idx10.append([])
    tmp = tficf[four_classes[i]].argsort()[::-1][:10]
    for j in range(10):
        idx10[i].append(vectorizer_c.get_feature_names()[tmp[j]])

for i in range(4):
    print ('10 most significant terms in ' + classes_4[i] + ' are: ')
    print (idx10[i])

print ('-'*30)
'''
#question d)
#applying LSI
SVD = TruncatedSVD(n_components = 50, algorithm = 'arpack')
train_LSI = SVD.fit_transform(train_tfidf)
test_LSI = SVD.transform(test_tfidf)

#applying NMF
nmf = NMF(n_components = 50, init = 'random', random_state = 0)
train_NMF = nmf.fit_transform(train_tfidf)
test_NMF = nmf.transform(test_tfidf)

#question e)
print ('doing SVM...')
#put 8 subclasses into 2 classes
label_train = []
label_test = []
two_classes = ['comp', 'rec']

for i in range(len(twenty_train.data)):
    if 'comp' in twenty_train.filenames[i]:
        label_train.append(0)
    else:
        label_train.append(1)
        
for i in range(len(twenty_test.data)):
    if 'comp' in twenty_test.filenames[i]:
        label_test.append(0)
    else:
        label_test.append(1)

#data report and plot
def metrics_analysis(y_test, y_predict, p = 1, target_names = None):
    accuracy = accuracy_score(y_test, y_predict)
    if p:
        print ("Confusion matrix:")
        print (confusion_matrix(y_test, y_predict))
        print ("Accuracy: " + str(accuracy))
        print ("Classification Report:")
        print (classification_report(y_test, y_predict, target_names = target_names))
    else:
        return accuracy

#ROC curve
def plot_roc_curve(y_test, scores, clf_name):
    if len(scores.shape) == 2:
        fpr, tpr, thresholds = roc_curve(y_test, scores[:,1])
    else:
        fpr, tpr, thresholds = roc_curve(y_test, scores)
    fig, ax = plt.subplots()
    roc_auc = auc(fpr,tpr)
    ax.plot(fpr, tpr, lw = 2, label = 'area under curve = %0.4f' % roc_auc)
    ax.grid(color = '0.7', linestyle = '--', linewidth = 1)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize = 15)
    ax.set_ylabel('True Positive Rate',fontsize = 15)
    plt.title('ROC_%s' %clf_name)
    ax.legend(loc = "lower right")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    plt.show()

def linsvm(gamma, method, data, label_train, test, label_test, p = 1, target_names = None):
    clf = svm.LinearSVC(C = gamma, random_state = 42)
    clf.fit(data, label_train)
    pred = clf.predict(test)  
    if p:
        print ("These are " + method + " SVM results with gamma = " + str(gamma))
        metrics_analysis(label_test, pred, target_names = target_names)
        print ('-'*30)
        scores = clf.decision_function(test)
        plot_roc_curve(label_test, scores,  method + " SVM, gamma = " + str(gamma) + ', min_df = ' + str(min_df))
    else:
        accuracy = metrics_analysis(label_test, pred, p = 0)
        return accuracy
'''
linsvm(1000, 'LSI', train_LSI, label_train, test_LSI, label_test, target_names = two_classes)
linsvm(1000, 'NMF', train_NMF, label_train, test_NMF, label_test, target_names = two_classes)
linsvm(0.001, 'LSI', train_LSI, label_train, test_LSI, label_test, target_names = two_classes)
linsvm(0.001, 'NMF', train_NMF, label_train, test_NMF, label_test, target_names = two_classes)

#question f)
print ('cross-validating...')
#creating the dataset folds for cross-valdiation
num_folds = 5
x_folds = []
y_folds = []
num = train_tfidf.shape[0]

for i in range(num_folds):
    x_folds.append(train_tfidf.toarray()[int(i*num/num_folds):int((i+1)*num/num_folds)])
    y_folds.append(label_train[int(i*num/num_folds):int((i+1)*num/num_folds)])

#cross-validation
gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
methods = ['LSI', 'NMF']

#find the best gamma based on accuracy
for method in methods:
    accuracy_gammas = []
    for gamma in gammas:
        accuracy_eachfold = []
        for i in range(num_folds):
            x_v = x_folds[i]
            y_v = y_folds[i]
            y_t = []
            if i == 0:
                x_t = x_folds[1][:]
                y_t.extend(y_folds[1])
                for j in range(2, num_folds):
                    x_t = np.vstack((x_t, x_folds[j]))
                    y_t.extend(y_folds[j])
            else:
                x_t = x_folds[0][:]
                y_t.extend(y_folds[0])
                for j in range(1, num_folds):
                    if j != i:
                        x_t = np.vstack((x_t, x_folds[j]))
                        y_t.extend(y_folds[j])
            accuracy = linsvm(gamma, method, x_t, y_t, x_v, y_v, p = 0)
            accuracy_eachfold.append(accuracy)
        accuracy_gammas.append(np.sum(accuracy_eachfold)/num_folds)
    print ('Using ' + method + ' SVM, here are the results of accuracy corresponding to gamma:')
    for i in range(len(accuracy_gammas)):
        print ('Gamma = ' + str(gammas[i]).rjust(6) + ': ' + 'accuracy = ' + str(accuracy_gammas[i]))   

print ('-'*30)
'''
#question g)
print ('doing Naive Bayes...') 
#using MultinomialNB for NMF (since Naive Bayes only takes non-negative values)
clf_nb = MultinomialNB()
clf_nb.fit(train_NMF, label_train)          
pred_nb_NMF = clf_nb.predict(test_NMF)      
scores_nb = clf_nb.predict_proba(test_NMF)

print ('These are the reults of MultinomialNB')
metrics_analysis(label_test, pred_nb_NMF, target_names = two_classes)
plot_roc_curve(label_test, scores_nb, "MultinomialNB")
print ('-'*30)

#question h)
print ('doing Logistic Regression...')
clf_lr = linear_model.LogisticRegression()
#LSI part
clf_lr.fit(train_LSI, label_train)
pred_lr_LSI = clf_lr.predict(test_LSI)
scores_lr_LSI = clf_lr.predict_proba(test_LSI)
print ('These are the reults of LSI logistic regression')
metrics_analysis(label_test, pred_lr_LSI, target_names = two_classes)
plot_roc_curve(label_test, scores_lr_LSI, "LogisticRegression_no-regularization_LSI")

#NMF part
clf_lr.fit(train_NMF, label_train)
pred_lr_NMF = clf_lr.predict(test_NMF)
scores_lr_NMF = clf_lr.predict_proba(test_NMF)
print ('These are the reults of NMF logistic regression')
metrics_analysis(label_test, pred_lr_NMF, target_names = two_classes)
plot_roc_curve(label_test, scores_lr_NMF, "LogisticRegression_no-regularization_NMF")

print ('-'*30)

#question i)
print ('doing Logistic Regression with regularization terms...')
reg_coef = [0.001,0.1,10,1000,100000]

#LSI part
print ("LogisticRegression LSI penalty = L1")
for coef in reg_coef:
    clf_lr1 = linear_model.LogisticRegression(C = coef, penalty = 'l1')
    clf_lr1.fit(train_LSI, label_train)
    pred_lr_LSI1 = clf_lr1.predict(test_LSI)
    scores_lr_LSI1 = clf_lr1.predict_proba(test_LSI)
    print ('coefficient:', coef)
    metrics_analysis(label_test, pred_lr_LSI1, target_names = two_classes)
    plot_roc_curve(label_test, scores_lr_LSI1, "LogisticRegression_L1_LSI with coefficient = " + str(coef))
    print ('-'*30)
print ("LogisticRegression LSI penalty = L2")
for coef in reg_coef:
    clf_lr2 = linear_model.LogisticRegression(C = coef, penalty = 'l2')
    clf_lr2.fit(train_LSI, label_train)
    pred_lr_LSI2 = clf_lr2.predict(test_LSI)
    scores_lr_LSI2 = clf_lr2.predict_proba(test_LSI)
    print ('coefficient:', coef)
    metrics_analysis(label_test, pred_lr_LSI2, target_names = two_classes)
    plot_roc_curve(label_test, scores_lr_LSI2, "LogisticRegression_L2_LSI with coefficient = " + str(coef))
    print ('-'*30)
#NMF part
print ("LogisticRegression NMF penalty = L1")
for coef in reg_coef:
    clf_lr1 = linear_model.LogisticRegression(C = coef, penalty = 'l1')
    clf_lr1.fit(train_NMF, label_train)
    pred_lr_NMF1 = clf_lr1.predict(test_NMF)
    scores_lr_NMF1 = clf_lr1.predict_proba(test_NMF)
    print ('coefficient:', coef)
    metrics_analysis(label_test, pred_lr_NMF1, target_names = two_classes)
    plot_roc_curve(label_test, scores_lr_NMF1, "LogisticRegression_L1_NMF with coefficient = " + str(coef))
    print ('-'*30)
print ("LogisticRegression NMF penalty = L2")
for coef in reg_coef:
    clf_lr2 = linear_model.LogisticRegression(C = coef, penalty = 'l2')
    clf_lr2.fit(train_NMF, label_train)
    pred_lr_NMF2 = clf_lr2.predict(test_NMF)
    scores_lr_NMF2 = clf_lr2.predict_proba(test_NMF)
    print ('coefficient:', coef)
    metrics_analysis(label_test, pred_lr_NMF2, target_names = two_classes)
    plot_roc_curve(label_test, scores_lr_NMF2, "LogisticRegression_L2_NMF with coefficient = " + str(coef))
    print ('-'*30)

# question i) Multiclass Classification
print ('doing Multiclass Classification...')
categories_i = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
twenty_train_i = fetch_20newsgroups(subset = 'train', categories = categories_i, shuffle = True, random_state = 42)
twenty_test_i = fetch_20newsgroups(subset = 'test', categories = categories_i, shuffle = True, random_state = 42)

for i in range(len(twenty_train_i.data)):
    twenty_train_i.data[i] = ''.join([s for s in twenty_train_i.data[i] if s.isalpha() or s.isspace() or s in punctuation])
    for ch in punctuation:
        twenty_train_i.data[i] = twenty_train_i.data[i].replace(ch, ' ')
for i in range(len(twenty_test_i.data)):
    twenty_test_i.data[i] = ''.join([s for s in twenty_test_i.data[i] if s.isalpha() or s.isspace() or s in punctuation])
    for ch in punctuation:
        twenty_test_i.data[i] = twenty_test_i.data[i].replace(ch, ' ')

for i in range(len(twenty_train_i.data)):
    words = twenty_train_i.data[i].split()
    for j in range(len(words)):
        words[j] = sbs.stem(words[j])
    twenty_train_i.data[i] = str(words)
for i in range(len(twenty_test_i.data)):
    words = twenty_test_i.data[i].split()
    for j in range(len(words)):
        words[j] = sbs.stem(words[j])
    twenty_test_i.data[i] = str(words)
    
train_count_i = vectorizer_count.fit_transform(twenty_train_i.data)
test_count_i = vectorizer_count.transform(twenty_test_i.data)
train_tfidf_i = tfidf_transformer.fit_transform(train_count_i)
test_tfidf_i = tfidf_transformer.transform(test_count_i)
train_LSI_i = SVD.fit_transform(train_tfidf_i)
test_LSI_i = SVD.transform(test_tfidf_i)
train_NMF_i = nmf.fit_transform(train_tfidf_i)
test_NMF_i = nmf.transform(test_tfidf_i)

# NaiveBayes
# LSI part
clf_nb_i_1 = GaussianNB()
clf_nb_i_1.fit(train_LSI_i, twenty_train_i.target)
pred_nb_LSI_i_1 = clf_nb_i_1.predict(test_LSI_i)
print ("Multiclass Naive Bayes LSI")
metrics_analysis(twenty_test_i.target, pred_nb_LSI_i_1, twenty_test_i.target_names)

# NMF part
clf_nb_i_2 = MultinomialNB()
clf_nb_i_2.fit(train_NMF_i, twenty_train_i.target)
pred_nb_NMF_i_2 = clf_nb_i_2.predict(test_NMF_i)
print ("Multiclass Naive Bayes NMF")
metrics_analysis(twenty_test_i.target, pred_nb_NMF_i_2, twenty_test_i.target_names)
'''
#SVM
#LSI part
print ("One vs One using LSI")
clf_1v1_LSI = OneVsOneClassifier(svm.LinearSVC(random_state = 42, class_weight = 'balanced'))
clf_1v1_LSI.fit(train_LSI_i, twenty_train_i.target)
pred_svm_LSI_1v1 = clf_1v1_LSI.predict(test_LSI_i)
metrics_analysis(twenty_test_i.target, pred_svm_LSI_1v1, target_names = twenty_test_i.target_names)
print ('-'*30)

print ("One vs Rest using LSI")
clf_1vr_LSI = OneVsRestClassifier(svm.LinearSVC(random_state = 42, class_weight = 'balanced'))
clf_1vr_LSI.fit(train_LSI_i, twenty_train_i.target)
pred_svm_LSI_1vr = clf_1vr_LSI.predict(test_LSI_i)
metrics_analysis(twenty_test_i.target, pred_svm_LSI_1vr, target_names = twenty_test_i.target_names)
print ('-'*30)

#NMF part
print ("One vs One using NMF")
clf_1v1_NMF = OneVsOneClassifier(svm.LinearSVC(random_state = 42, class_weight = 'balanced'))
clf_1v1_NMF.fit(train_NMF_i, twenty_train_i.target)
pred_svm_NMF_1v1 = clf_1v1_NMF.predict(test_NMF_i)
metrics_analysis(twenty_test_i.target, pred_svm_NMF_1v1, target_names = twenty_test_i.target_names)
print ('-'*30)

print ("One vs Rest using NMF")
clf_1vr_NMF = OneVsRestClassifier(svm.LinearSVC(random_state = 42, class_weight = 'balanced'))
clf_1vr_NMF.fit(train_NMF_i, twenty_train_i.target)
pred_svm_NMF_1vr = clf_1vr_NMF.predict(test_NMF_i)
metrics_analysis(twenty_test_i.target, pred_svm_NMF_1vr, target_names = twenty_test_i.target_names)
'''
print ('It\'s Done!')