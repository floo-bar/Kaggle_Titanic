import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
train=pd.read_csv('train.csv')
Y=train.iloc[:,1].values
test=pd.read_csv('test.csv')
train.head()
test.head()

#Drop unecessary information
train=train.drop(['Survived','Cabin','Embarked','Name','Ticket'],axis=1).values 

#Encoding gender
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_tr=LabelEncoder()
train[:,2]=labelencoder_tr.fit_transform(train[:,2])

#Divide into training and testing data
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(train,Y,test_size=0.33,random_state=42)

#Decision Tree Classifier-0.78
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print('Training=',accuracy_score(Y_train,classifier.predict(X_train)))
print('Validation=',accuracy_score(Y_test,classifier.predict(X_test)))


#Support Vector Machine -0.80
from sklearn.svm import SVC
classifier2=SVC(kernel='linear')
classifier2.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
print('Training=',accuracy_score(Y_train,classifier.predict(X_train)))
print('Validation=',accuracy_score(Y_test,classifier.predict(X_test)))

#Test file
test=test.drop(['Cabin','Embarked','Name','Ticket'],axis=1).values
labelencoder_te=LabelEncoder()
test[:,2]=labelencoder_te.fit_transform(test[:,2])

#Fill in NaN data
test['Age'] = test['Age'].fillna(np.mean(test['Age']))
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

#Predict for test data
Y_new=classifier2.predict(test)
