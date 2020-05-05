#ITI86600 Homework 4
#@authors Guillaume Ricard and Noel Schneider
#December 2017

import pandas as pd #manipulate and analyze datasets
import numpy as np #manipulate tables and matrix, do operations
import matplotlib.pyplot as plt #trace and visualize data with graphs
import math
import sklearn #machine learning

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#statistics
from scipy import stats
from scipy import misc

import io
import pydot

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_digits

from os import listdir
from os.path import isfile, join

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn import neighbors


# In[7]:


def cutWidth(dataset, i, j):
    split1 = dataset[dataset.x > i]
    split2 = split1[split1.x < j]
    return split2

def cutHeight(dataset, i, j):
    split1 = dataset[dataset.y > i]
    split2 = split1[split1.y < j]
    return split2

def SplitInDigits(initRangeReal, endRangeReal, dataset):
    found = 0
    endRange = int(endRangeReal)
    initRange = int(initRangeReal)
    end = True

    while end:
        initRange = initRange + 1
        if initRange > endRange:
            return 0, 0
        for i in range(initRange, endRange):
            split = cutWidth(dataset,i,i+5)
            digit = cutWidth(dataset, initRange, i)
            if split.size < 5 and digit.size > 50:
                found = i + 5
                end = False
                break

        if found == 0:
            end = True

    return initRange, found



# In[22]:


def cleanDataset(arrayDataset): #remove noise points
    for k in range(0, len(arrayDataset)):
        dset = arrayDataset[k]
        RightEdgeX = dset.x.max()
        LeftEdgeX = dset.x.min()
        for i in range(int(LeftEdgeX), int(LeftEdgeX + (RightEdgeX - LeftEdgeX) / 2)):
            hSplit = cutWidth(dset,i,i+3)
            if hSplit.size == 0:
                arrayDataset[k] = cutWidth(dset, i+3, RightEdgeX)



# In[26]:


def prepareTestSet(fileName):
    usefulData = pd.read_csv(fileName) #unused columns are removed
    usefulData = usefulData[['x','y']]
    usefulData['y'] = usefulData['y'].apply(lambda x: -x)
    usefulData.plot.scatter(x='x', y='y')

    RightEdgeX = usefulData.x.max()

    init = 10
    arrayDataset = []
    for i in range(0,9):
        init, index = SplitInDigits(init, RightEdgeX, usefulData)
        if init == 0 and index == 0:
            #print ("Error when reading image")
            break

        tmp2 = cutWidth(usefulData, init, index)
        arrayDataset.append(tmp2)
        init = index + 7

    last = usefulData[usefulData.x > init]
    arrayDataset.append(last)

    cleanDataset(arrayDataset)

    for dset in arrayDataset:
        dset.plot.scatter(x='x', y='y')

    for i in range(0,len(arrayDataset)):
        t = arrayDataset[i]

        Xmean = t['x'].mean()
        Xstd = t['x'].std()

        Ymean = t['y'].mean()
        Ystd = t['y'].std()

    #DIVISION in an 8x8 grid
    matrixList = []
    for digit in arrayDataset:
        matrix = []

        Ymin = digit['y'].min()
        Ymax = digit['y'].max()
        Ystep = (Ymax - Ymin) / 8

        Xmin = digit['x'].min()
        Xmax = digit['x'].max()
        Xstep = (Xmax - Xmin) / 6

        for j in range(7,-1,-1):
            square = cutHeight(digit, Ymin + (j*Ystep), Ymin + ((j+1)* Ystep))

            tmp = []
            tmp.append(0)
            for i in range(0,6):
                fraction = cutWidth(square, Xmin + (i*Xstep), Xmin + ((i+1)* Xstep))

                tmp.append(fraction.size)
            tmp.append(0)

            matrix.append(tmp)

        matrixList.append(matrix)

    for matrix in matrixList:
        for array in matrix:
            for i in range(0,8):
                if array[i] != 0:
                    array[i] = 16

    columns = []
    for i in range(0,64):
        columns.append("X" + str(i))
    columns.append("target")

    i = 0
    tmp = []
    for m in matrixList:
        tmpColumns = np.asarray(m).reshape(-1)
        addTarget = np.append(tmpColumns,[i])
        i= i + 1
        tmp.append(addTarget)
    return pd.DataFrame(tmp, columns = columns)



# In[27]:


digits = load_digits() #training database from SKLearn
tri = prepareTestSet('handwrittendigits/mag3_EE6653C8_digits_2017-12-08_14_18_28___0444df099e184398993a95d0f40446fe.csv')


# In[28]:


collectedFiles = [f for f in listdir('handwrittendigits/') if isfile(join('handwrittendigits/', f))]
file = collectedFiles.pop(0)
trainingData =  prepareTestSet('handwrittendigits/' + file)
i = 0
for file in reversed(collectedFiles):
    print ('\n' + file + '\n')
    trainingData =  pd.concat([trainingData, prepareTestSet('handwrittendigits/' + file)])
    print (str(i+1) + ' of ' + str(len(collectedFiles)))
    i = i + 1


# In[29]:


trainingData.to_csv('trainingData.csv')


# In[44]:


trained_Data, test = train_test_split(trainingData, test_size = 0.25)


# In[45]:


#TRAINING SET
#Database converted in Pandas

columns = []
for i in range(0,64):
    columns.append("X" + str(i))
columns.append("target")
tmp = []

i = 0
for img in digits.images:
    tmpColumns = np.asarray(img).reshape(-1)
    addTarget = np.append(tmpColumns,[digits.target[i]])
    i = i + 1

    tmp.append(addTarget)

training = pd.DataFrame(tmp, columns = columns)

training = pd.concat([training, trained_Data])


# In[32]:


# Plot image training set
def printImage(dataset):
    printImage = dataset.sample(n = 8)
    index = 0
    for img in printImage.as_matrix():
        end = []
        for i in range(0, 8):
            tmp = []
            for j in range(0, 8):
                tmp.append(img[i * 8 + j])
            end.append(tmp)
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(end, cmap=plt.cm.gray_r, interpolation='nearest', shape=(8,8))
        plt.title('%i' % img[64])
        index = index + 1
    plt.show()


# In[46]:


print("Training:")
printImage(training)
print("Testing:")
printImage(test)


# In[48]:


target = 'target'
removed = ['X0', 'X7',
           'X8', 'X15',
           'X16', 'X23',
           'X24', 'X31',
           'X32', 'X39',
           'X40', 'X47',
           'X48', 'X55',
           'X56', 'X63',
           target]
tempds = training.drop(removed, axis=1)
label = tempds.columns

values = training[label]

x = values
y = training['target']

for col in label:
    training[col] = training[col].apply(lambda x: 0 if x < 8  else 16)
training.head()
#printImage(training)
#printImage(test)

x = training[label]
y = training[target]

simple_logistic_separate = RandomForestClassifier(n_estimators=1500, max_depth=None,
                            min_samples_split=5, n_jobs=-1)
simple_logistic_separate.fit(x,y)

pd.concat((pd.DataFrame(label, columns = ['variable']),
                   pd.DataFrame(simple_logistic_separate.feature_importances_, columns = ['importance'])),
                   axis = 1).sort_values(by='importance', ascending = False)[:200]


# In[49]:


train_separate, test_separate = train_test_split(training, test_size = 0.2)

print ("Train set: ", len(train_separate))
print ("Test set: ", len(test_separate))



# In[51]:


x = train_separate[label]
y = train_separate[target]

simple_logistic_separate = RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=2, random_state=0)
simple_logistic_separate.fit(x,y)

pd.concat((pd.DataFrame(label, columns = ['variable']),
                   pd.DataFrame(simple_logistic_separate.feature_importances_, columns = ['importance'])),
                   axis = 1).sort_values(by='importance', ascending = False)[:200]



# In[52]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def accuracyModel(test_separate, model):
    y_true = test_separate[target]
    x_true = test_separate[label]
    yp = model.predict(x_true)
    yprob = model.predict_proba(x_true)

    accuracy = accuracy_score(y_true, yp)
    print ("accuracy: ", accuracy)

    cm = confusion_matrix(test_separate[target],yp)
    print (cm)
    return (yp)


# In[53]:


accuracyModel(test_separate, simple_logistic_separate)
test['predicted'] = accuracyModel(test, simple_logistic_separate)


# In[54]:


test.plot.scatter(x='predicted', y='target')


# In[59]:


classifiers = [
       neighbors.KNeighborsClassifier(20, weights='distance', n_jobs = -1),
       linear_model.LogisticRegression(C=10e10, penalty='l1' ,n_jobs = -1),
       RandomForestClassifier(n_estimators=1000, min_samples_split=5, max_depth=None, n_jobs=-1),
       DecisionTreeClassifier('gini')]

label_columns = ["Classifier", "Accuracy"]
label = pd.DataFrame(columns=label_columns)

nFolds = 8
shuffleSplit = StratifiedShuffleSplit(n_splits=nFolds, test_size=0.25, random_state=3)

X = training[label]
y = training[target]

accuracy_dict = {}
count = 0

for train_index, test_index in shuffleSplit.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for clf in classifiers:
        #print(count)
        count = count + 1
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += accuracy
        else:
            acc_dict[name] = accuracy
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / number_of_folds
    print(clf,accuracy_dict[clf])
    log_entry = pd.DataFrame([[clf, accuracy_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', data=log, color="r")
plt.figure()
