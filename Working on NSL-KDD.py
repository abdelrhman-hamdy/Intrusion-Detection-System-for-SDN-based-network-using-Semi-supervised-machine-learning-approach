# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
#import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

#------------------------------------------------------------------------
handle = open('KDDTrain+_20Percent.arff')
lis=[]
list1=[]
for line in handle :
    if line.startswith('@attribute'):
         lis.append(line.split(' ')[1])
for x in lis : 
    s=x.split("'")[1]
    list1.append(s) 
list1.append('difficulties')

#------------------------------------------------------------------------
#DDoS Attack only data
train = pd.read_csv('data.csv')
train.columns=list1
ddos_list=['back','land','neptune','pod','smurf','teardrop','mailbomb','processtable','udpstorm','apache2','worm','normal']
train['d']=train['class'].apply(lambda x : 1 if x in ddos_list  else 0)
train=train.query('d == 1')
train.drop('d',axis=1,inplace=True)
train['class']=train['class'].apply(lambda x : 0 if x =='normal' else 1)

#------------------------------------------------------------------------

test = pd.read_csv('train.csv')
test.columns =list1
ddos_list=['back','land','neptune','pod','smurf','teardrop','mailbomb','processtable','udpstorm','apache2','worm','normal']
test['d']=test['class'].apply(lambda x : 1 if x in ddos_list  else 0)
test=test.query('d == 1')
test.drop('d',axis=1,inplace=True)
test['class']=test['class'].apply(lambda x : 0 if x =='normal' else 1)
test.drop(12730,inplace=True)
#----------------------------------------------------------------------------

test = train.iloc[:,[0,1,3,4]]
#x=test.query('protocol_type == "icmp"').loc[:,'protocol_type'].index‏‏
x=test.query('protocol_type == 1').loc[:,'protocol_type'].index
#‏
train.protocol_type.replace(1,'icmp',inplace = True)
train.protocol_type.replace(6,'tcp',inplace = True)
train.protocol_type.replace(17,'udp',inplace = True)
#------------------------------------------------------------------------
#flag label encoding
col =[]
from sklearn.preprocessing import LabelEncoder
for column in ['flag'] :
    le = LabelEncoder()
    le.fit(train[column])
    train[column]=le.transform(train[column])
    test[column]=le.transform(test[column])
  
#------------------------------------------------------------------------
#Label encoding service
t=list(train.service.unique())
for x in list(test.service.unique()): 
        # check if exists in unique_list or not 
        if x not in t: 
            t.append(x) 
dicit ={'service_enc':np.arange(0,len(t)),'service':t}
enc=pd.DataFrame(dicit)
train = pd.merge(train,enc,on='service',how='left')
train['service'] =train['service_enc']
train.drop('service_enc',axis=1,inplace=True)
#test['class'] = pd.get_dummies(test['class']).iloc[:,0]
test=pd.merge(test,enc,on='service',how='left')
test['service'] =test['service_enc']
test.drop('service_enc',axis=1,inplace=True)

#------------------------------------------------------------------------
#one-hot encoder
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

train[['icmp','tcp','udp']] = pd.get_dummies(train.protocol_type)
train.drop('protocol_type',axis=1,inplace=True)
train.head()

test[['icmp','tcp','udp']] = pd.get_dummies(test.protocol_type)
test.drop('protocol_type',axis=1,inplace=True)
test.head()
#------------------------------------------------------------------------
#Splitting train and test results

X_train = train.drop('class',axis=1)
y_train =train['class']

X_test = test.drop('class',axis=1)
y_test =test['class']

#------------------------------------------------------------------------
#Normalization

from sklearn.preprocessing import MinMaxScaler
for i in X_train.columns[:-3]:
    
    # fit on training data column
    scale = MinMaxScaler().fit(X_train[[i]])
    
    # transform the training data column
    X_train[i] = scale.transform(X_train[[i]])
    
    # transform the testing data column
    X_test[i] = scale.transform(X_test[[i]])

#------------------------------------------------------------------------
#Standardization
from sklearn.preprocessing import StandardScaler

for i in X_train.columns[:-3]:
    
    # fit on training data column
    scale = StandardScaler().fit(X_train[[i]])
    
    # transform the training data column
    X_train[i] = scale.transform(X_train[[i]])
    
    # transform the testing data column
    X_test[i] = scale.transform(X_test[[i]])

#------------------------------------------------------------------------
#Train Accuracy :  99.45439617194467 % Standardization
#Test Accuracy :  92.34667132622751 % Standardization

#Train Accuracy :  99.45086475558185 % Normalization
#Test Accuracy :  92.0379754208166 % Normalization
#Random Forest Classifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0 )
classifier.fit(X_train , y_train)

print("Train Accuracy : ",classifier.score(X_train, y_train) *100,"%")
print("Test Accuracy : ",classifier.score(X_test, y_test) *100,"%")
#------------------------------------------------------------------------

y_pred = classifier.predict(X_test) 

cm = confusion_matrix(y_test, y_pred)
print(cm)

#------------------------------------------------------------------------
#Train Accuracy :  96.0942535027236 % Standardization
#Test Accuracy :  90.61680936571727 % Standardization

#Train Accuracy :  94.95095745526136 % Normalization
#Test Accuracy :  91.54872153299551 % Normalization
# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Train Accuracy : ",classifier.score(X_train, y_train) *100,"%")
print("Test Accuracy : ",classifier.score(X_test, y_test) *100,"%")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#------------------------------------------------------------------------
#Train Accuracy :  99.43497338194916 % Standardization
#Test Accuracy :  93.20869008095987 % Standardization

#Train Accuracy :  99.42349627877 % Normalization
#Test Accuracy :  91.59531714135942 % Normalization
#GradientBoost
classifier=GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=0)           
classifier.fit(X_train,y_train)          

print("Train Accuracy : ",classifier.score(X_train, y_train) *100,"%")
print("Test Accuracy : ",classifier.score(X_test, y_test) *100,"%")


y_pred = classifier.predict(X_test) 
cm = confusion_matrix(y_test, y_pred)
print(cm)

#------------------------------------------------------------------------
#Train Accuracy :  99.43320767376775 % Standradization
#Test Accuracy :  87.04642087483253 % Standradization

#Train Accuracy :  99.44645048512832 % Normalization
#Test Accuracy :  89.90040188712214 % Normalization

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Train Accuracy : ",classifier.score(X_train, y_train) *100,"%")
print("Test Accuracy : ",classifier.score(X_test, y_test) *100,"%")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#------------------------------------------------------------------------
#Train Accuracy :  99.45439617194467 % standradization
#Test Accuracy :  93.138796668414 % standradization

#Train Accuracy :  99.45086475558185 % Normalization
#Test Accuracy :  92.32337352204554 % Normalization

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Train Accuracy : ",classifier.score(X_train, y_train) *100,"%")
print("Test Accuracy : ",classifier.score(X_test, y_test) *100,"%")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#------------------------------------------------------------------------
#Train Accuracy :  94.08840900864314 % Standardization
#Test Accuracy :  88.5491292445687 % Standardization

#Train Accuracy :  93.99041220457495 % Normalization
#Test Accuracy :  91.292445686994 % Normalization

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Train Accuracy : ",classifier.score(X_train, y_train) *100,"%")
print("Test Accuracy : ",classifier.score(X_test, y_test) *100,"%")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#------------------------------------------------------------------------
#Train Accuracy :  54.0765787638277 % Standardization
#Test Accuracy :  47.638185101054226 % Standardization

#Train Accuracy :  53.57953191076111 % Normalization
#Test Accuracy :  47.42850486341662 % Normalization
# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Train Accuracy : ",classifier.score(X_train, y_train) *100,"%")
print("Test Accuracy : ",classifier.score(X_test, y_test) *100,"%")

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#------------------------------------------------------------------------
#Train Accuracy :  94.41 % Standardization
#Test Accuracy : 88.61 % Standardization

#Train Accuracy :  94.38 % Normalization
#Test Accuracy :  91.40 % Normalization
#Keras DL
import keras
from keras import layers
from keras import Model
from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import regularizers


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,7)))
model.add(tf.keras.layers.Dense(5, activation='relu',activity_regularizer=regularizers.l1(1e-4)))
model.add(tf.keras.layers.Dense(4, activation='relu',activity_regularizer=regularizers.l1(1e-4)))  
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='mean_absolute_error',metrics=["accuracy"])
model.fit(X_train,y_train , epochs=20, batch_size = 128)






#model = tf.keras.Sequential([keras.layers.Dense(units=3,input_shape=[7])])
#model.compile(optimizer='sgd', loss='mean_squared_error')
#model.fit(X_train,y_train , epochs=20)

predict = model.predict(X_test)
acc_arr = []

for i in predict:
    if i > 0.5:
        acc_arr.append(1)
    else:
        acc_arr.append(0)
            
        
cm = confusion_matrix(y_test, acc_arr)
print(cm)
model.evaluate(X_test, y_test)
#------------------------------------------------------------------------


