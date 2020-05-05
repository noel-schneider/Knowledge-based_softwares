
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# stat libraries
from scipy import stats
from scipy import misc

# Libraries for the evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# libraries needed to plot decision trees
import io
import pydot

get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[3]:


def cutDataset(dataset, i, j):
    cutted = dataset[dataset.x > i]
    cutted2 = cutted[cutted.x < j]
    #print(cutted2.describe())
    return cutted2

def cutDatasetY(dataset, i, j):
    cutted = dataset[dataset.y > i]
    cutted2 = cutted[cutted.y < j]
    #print(cutted2.describe())
    return cutted2

def makeTheCut(initRangeReal, endRangeReal, dataset):
    found = 0
    endRange = int(endRangeReal)
    initRange = int(initRangeReal)
    end = True
    #print(dataset.describe())
    while end:
        initRange = initRange + 1
        #print 'init ' + str(initRange)
        #print 'end ' + str(endRange)
        if initRange > endRange:
            return 0, 0
        for i in range(initRange, endRange):
            cutted2 = cutDataset(dataset,i,i+5)
            digit = cutDataset(dataset, initRange, i)
            if cutted2.size < 5 and digit.size > 50:
                found = i + 5
                end = False
                #print '-----FOUND------'
                break
                
        if found == 0:
            end = True
        
    return initRange, found


# In[4]:


def removeNoise(datasetArray):
    for k in range(0, len(datasetArray)):
        ds = datasetArray[k]
        maxX = ds.x.max()
        minX = ds.x.min()
        #print 'min ' + str(minX) + ' max ' + str(maxX)
        for i in range(int(minX), int(minX + (maxX - minX) / 2)):
            cutted2 = cutDataset(ds,i,i+3)
            #print cutted2.size
            if cutted2.size == 0:
                #print 'trovato: ' + str(i)
                datasetArray[k] = cutDataset(ds, i+3, maxX)
        #datasetArray[k].plot.scatter(x='x', y='y')


# In[5]:


def preprocessingTest(fileName):
    realTest = pd.read_csv(fileName) #Read File and removed unused columns
    realTest = realTest[['x','y']]
    realTest['y'] = realTest['y'].apply(lambda x: -x)
    realTest.plot.scatter(x='x', y='y')
    
    maxX = realTest.x.max() #cut the dataset in different digits
    
    init = 10
    datasetArray = []
    for i in range(0,9):
        init, index = makeTheCut(init, maxX, realTest)
        if init == 0 and index == 0:
            print 'can not read the image'
            break
        temp2 = cutDataset(realTest, init, index)

        #temp2.plot.scatter(x='x', y='y')

        datasetArray.append(temp2)

        #print datasetArray[i].describe()

        #datasetArray[i].plot.scatter(x='x', y='y')
        init = index + 7
        #print i

    last = realTest[realTest.x > init]
    #last.plot.scatter(x='x', y='y')
    datasetArray.append(last)
    
    removeNoise(datasetArray)
    
    for ds in datasetArray:
        ds.plot.scatter(x='x', y='y')
    
    for i in range(0,len(datasetArray)): #Standardize X by (x-mean)/std^2
        t = datasetArray[i]

        meanX = t['x'].mean()
        stdX = t['x'].std()

        #t.loc[:, 'x'] = ( t['x'] -  meanX) / stdX

        meanY = t['y'].mean()
        stdY = t['y'].std()

        #t.loc[:, 'y'] = ( t['y'] -  meanY) / stdY

        #t.plot.scatter(x='x', y='y')
        
        #try to divide the number in an 8x8 grid
    megamatrix = []
    for digit in datasetArray:
        matrix = []

        minimumX = digit['x'].min()
        maximumX = digit['x'].max()
        unitX = (maximumX - minimumX) / 6

        minimumY = digit['y'].min()
        maximumY = digit['y'].max()
        unitY = (maximumY - minimumY) / 8

        for j in range(7,-1,-1):
            square = cutDatasetY(digit, minimumY + (j*unitY), minimumY + ((j+1)* unitY))
                
            #fraction.plot.scatter(x='x', y='y')
            temp = []
            temp.append(0)
            for i in range(0,6):
                fraction = cutDataset(square, minimumX + (i*unitX), minimumX + ((i+1)* unitX))
            
                temp.append(fraction.size)
            temp.append(0)

            matrix.append(temp)

        #matrix = np.transpose(matrix)
        megamatrix.append(matrix)

    for matrix in megamatrix:
        for arr in matrix:    
            for i in range(0,8):
                if arr[i] != 0:
                    arr[i] = 16
            #print arr
        #print '\n'
            
    columns = []
    for i in range(0,64):
        columns.append("X" + str(i))   
    columns.append("target")

    i = 0
    temp = []
    for m in megamatrix:
        tempColumns = np.asarray(m).reshape(-1)
        addTarget = np.append(tempColumns,[i])
        i= i + 1
        temp.append(addTarget)
    return pd.DataFrame(temp, columns = columns)


# In[2]:


tri = preprocessingTest('test/cattivagente/mag20_02199244_digits_2017-12-08_14_30_52___75ad24c2110b45119811cd7f4bc1b6ad.csv')


# In[7]:


from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('test/') if isfile(join('test/', f))]
rf = onlyfiles.pop(0)
#print onlyfiles
ttuData =  preprocessingTest('test/'+rf)
i = 0
for rf in reversed(onlyfiles):
    print '\n' + rf + '\n'
    ttuData =  pd.concat([ttuData, preprocessingTest('test/'+rf)])
    print str(i) + ' of ' + str(len(onlyfiles))
    i = i + 1


# In[8]:


ttuData.to_csv('ttuData.csv')
ttuData.head()


# In[9]:


ttuData.describe()


# In[10]:


train_TTUData, test = train_test_split(ttuData, test_size = 0.2)


# In[11]:


test.head()


# In[12]:


test.describe()


# In[13]:


#WORK on the TRAINING SET
#CONVERT DATABASE IN PANDAS
#arr = []

columns = []
for i in range(0,64):
    columns.append("X" + str(i))   
columns.append("target")
temp = []

i = 0
for img in digits.images:
    tempColumns = np.asarray(img).reshape(-1)
    addTarget = np.append(tempColumns,[digits.target[i]])
    i = i + 1
    
    temp.append(addTarget)
    
train = pd.DataFrame(temp, columns = columns)

train = pd.concat([train, train_TTUData])
#train = train_TTUData
train.describe()


# In[14]:


# Plot image training set
def printImage(dataset):
    printImage = dataset.sample(n = 8)
    index = 0
    for img in printImage.as_matrix():
        end = []
        for i in range(0, 8):
            temp = []
            for j in range(0, 8):
                temp.append(img[i * 8 + j])
            end.append(temp)
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(end, cmap=plt.cm.gray_r, interpolation='nearest', shape=(8,8))
        plt.title('Training: %i' % img[64])
        index = index + 1
    plt.show()


# In[15]:


printImage(train)
printImage(test)


# *GENERO TANTI MODELLI E PER OGNI MODELLO CALCOLO UN PO COME FUNZIONA*

# In[16]:


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
tempds = train.drop(removed, axis=1)
label = tempds.columns

values = train[label]

x = values
y = train['target']


# In[17]:


for col in label:
    train[col] = train[col].apply(lambda x: 0 if x < 8  else 16)
train.head()
printImage(train)
printImage(test)


# In[18]:


x = train[label]
y = train[target]

simple_logistic_separate = RandomForestClassifier(n_estimators=1500, max_depth=None,
                            min_samples_split=5, n_jobs=-1)
simple_logistic_separate.fit(x,y)

pd.concat((pd.DataFrame(label, columns = ['variable']), 
                   pd.DataFrame(simple_logistic_separate.feature_importances_, columns = ['importance'])), 
                   axis = 1).sort_values(by='importance', ascending = False)[:200]


# In[19]:


train_separate, test_separate = train_test_split(train, test_size = 0.2)

print "Train set: ", len(train_separate)
print "Test set: ", len(test_separate)

train_separate.head()


# In[20]:


x = train_separate[label]
y = train_separate[target]

simple_logistic_separate = RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=2, random_state=0)
simple_logistic_separate.fit(x,y)


# In[24]:


def evaluateAccuracy(test_separate, model):
    y_true = test_separate[target]
    x_true = test_separate[label]
    yp = model.predict(x_true)
    yprob = model.predict_proba(x_true)
    
    accuracy = accuracy_score(y_true, yp)
    print ("accuracy: ", accuracy)
    
    cm = confusion_matrix(test_separate[target],yp)
    print cm
    return yp


# In[25]:


evaluateAccuracy(test_separate, simple_logistic_separate)
test['predicted'] = evaluateAccuracy(test, simple_logistic_separate)


# In[28]:


print test[['target', 'predicted']]


# In[ ]:


# def printImageP(dataset):
    printImage = dataset.sample(n = 8)
    index = 0
    for img in printImage.as_matrix():
        end = []
        for i in range(0, 8):
            temp = []
            for j in range(0, 8):
                temp.append(img[i * 8 + j])
            end.append(temp)
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(end, cmap=plt.cm.gray_r, interpolation='nearest', shape=(8,8))
        plt.title('Rl ' + str(img[64]) + " - Prted: " + str(img[65]))
        index = index + 1
    plt.show()


# In[59]:


printImageP(test[test.predicted != test.target])
test.plot.scatter(x='predicted', y='target')


# In[23]:


import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import linear_model
from sklearn import neighbors

classifiers = [
       neighbors.KNeighborsClassifier(20, weights='distance', n_jobs = -1),
       RandomForestClassifier(n_estimators=1000, min_samples_split=5, max_depth=None, n_jobs=-1),
       BaggingClassifier(n_estimators=50, n_jobs=-1),
       AdaBoostClassifier(n_estimators=500),
       DecisionTreeClassifier('gini'),
       linear_model.LogisticRegression(C=10e10, penalty='l1' ,n_jobs = -1)]

log_cols = ["Classifier", "acc"]
log = pd.DataFrame(columns=log_cols)

number_of_folds = 8
sss = StratifiedShuffleSplit(n_splits=number_of_folds, test_size=0.25, random_state=3)

X = train[label]
y = train[target]

acc_dict = {}
count = 0

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for clf in classifiers:
        #print(count)
        count = count + 1
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / number_of_folds
    print(clf,acc_dict[clf])
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)
plt.xlabel('acc')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='acc', y='Classifier', data=log, color="b")
plt.figure()

