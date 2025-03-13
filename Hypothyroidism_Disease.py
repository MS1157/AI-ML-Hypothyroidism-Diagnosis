#Require libraries.
import numpy as np 
import pandas as pd 
import sklearn as svm
from  sklearn.metrics import accuracy_score
from  sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import train_test_split
from  sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import seaborn as sns

#Importing our Datasets
dataset = pd.read_csv("Hypothyroidism_data.csv")

#Datafram Object as pandas
print(type(dataset))

#Shape of dataset
print(dataset.shape)

#Columns in Dataset
print(dataset.head(5))
print(dataset.sample(5))

#Description
print(dataset.describe())
print(dataset.info())

#Columns Understanding
info=["Age","Sex:1 = Male;0 = Female","On Thyroxine (1 = Yes; 0 = No)","TSH","T3 measured(pmol/L):1 = Yes; 0 = No","T3","TT4","BinaryClass:0= NO_Thyroid disease,1=thyroid disease "]
for i in range (len(info)):
  print(dataset.columns[i]+"\t\t", info[i])
  print(dataset['BinaryClass'].unique())

#Check Correlation
print(dataset.corr()['BinaryClass'].abs().sort_values(ascending=False))

#EDA  (Exploratory Data Analysis)
y=dataset['BinaryClass']
sns.catplot(y)
target_temp= dataset.BinaryClass.value_counts()


#Analysis Features
#Age
print(dataset['Age'].unique())
print(sns.catplot(x=dataset['Age'],y=dataset['BinaryClass']))
plt.show()

#SEX (FEMALE,MALE,NOT_DEFINE)
print(dataset['Sex'].unique())
print(sns.catplot(x=dataset["Sex"],y=dataset["BinaryClass"]))
plt.show()

#On Thyroxine(Y/N)
print(dataset['On thyroxine'].describe())
print(dataset['On thyroxine'].unique())
print(sns.catplot(data=dataset,x=dataset['On thyroxine'],y=dataset['BinaryClass']))
plt.show()

#TSH  Analysis (Numerical Data)
print(dataset['TSH'].describe())
print(dataset['TSH'].unique())
print(sns.catplot(data=dataset,x=dataset['TSH'],y=dataset['BinaryClass']))
plt.show()

#T3 Mesured Analysis
print(dataset['T3 measured'].describe())
print(dataset['T3 measured'].unique())
print(sns.catplot(data=dataset,x=dataset['T3 measured'],y=dataset['BinaryClass']))
plt.show()

#T3 Analysis (Numerical Data)
print(dataset['T3'].describe())
print(dataset['T3'].unique())
print(sns.catplot(data=dataset,x=dataset['T3'],y=dataset['BinaryClass']))
plt.show()

#T4 Analysis(Numerical Data)
print(dataset['TT4'].describe())
print(dataset['TT4'].unique())
print(sns.catplot(data=dataset,x=dataset['TT4'],y=dataset['BinaryClass']))
plt.show()

#//////////////////////////////////////////
#TRAIN TEST SPLIT
Features=dataset.drop(columns=['Age','BinaryClass'],axis=1)
Target=dataset['BinaryClass']
#SPLIT THE DATA
x_train,x_test,y_train,y_test=train_test_split(Features,Target,test_size=0.2,random_state=2)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

#Now Train the model
#Train model With Support Vector Machine(SVM)
SVM_model = SVC(kernel='linear')
print(SVM_model.fit(x_train,y_train))
#Predict the accuracy
#Train Data
SVM_x_train_prediction = SVM_model.predict(x_train)
SVM_Traindata_accuracy = accuracy_score(y_train,SVM_x_train_prediction)
print("The accuracy score achieved using Train data of SVM:", SVM_Traindata_accuracy)
#TEST DATA
SVM_x_test_prediction=SVM_model.predict(x_test)
SVM_data_accuracy= accuracy_score(y_test,SVM_x_test_prediction)
print("The accuracy score achieved using Test data of SVM :",SVM_data_accuracy)
#////////////////////////////
#Random Forest Train Model
RF_model =RandomForestClassifier(n_estimators=10,random_state=41)
RF_model.fit(x_train, y_train)
#Train data
RF_x_train_predict = RF_model.predict(x_train)
RF_train_data_accuracy = accuracy_score(y_train, RF_x_train_predict)
print("The accuracy score achieved using Train data of Random Forest",RF_train_data_accuracy)
# Test Data
RF_x_train_prediction= RF_model.predict(x_test)  # Corrected line: use RFmodel
RF_test_data_accuracy = accuracy_score(y_test,RF_x_train_prediction)
print("The accuracy score achieved using Test data of Random Forest:",RF_test_data_accuracy)
#//////////////////////////////
#Logistic Regression
lr_model=LogisticRegression()
lr_model.fit(x_train,y_train)
#Train Data
lr_x_train_prediction=lr_model.predict(x_train)
lr_train_data_accuracy=accuracy_score(y_train,lr_x_train_prediction)
print("The accuracy score achieved using Train data of Logistic Regression is:",lr_train_data_accuracy)
#Test Data
lr_x_train_prediction=lr_model.predict(x_test)
lr_test_data_accuracy=accuracy_score(y_test,lr_x_train_prediction)
print("The accuracy score achieved using  Test data of Logistic Regression is :",lr_test_data_accuracy)
 


#PREDICITION SYSTEM
Inputdata = ("Enter your Data")
Inputdata_numpy    = np.asarray(Inputdata)
Inputdata_reshaped = Inputdata_numpy.reshape(1,-1)
Prediction1        = SVM_model.predict(Inputdata_reshaped)
Prediction2        = RF_model.predict(Inputdata_reshaped)
Prediction3        = lr_model.predict(Inputdata_reshaped)
PREDICTION = [Prediction1,Prediction2,Prediction3]
print(PREDICTION)
if(PREDICTION[0]==0):
     print("The patient shows no signs of Hypothyroidism.")
else:
    print("The patient shows signs of Hypothyroidism.")    
