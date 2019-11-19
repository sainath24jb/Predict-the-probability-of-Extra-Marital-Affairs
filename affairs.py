# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:42:50 2019

@author: Hello
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

affair= pd.read_csv("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\logisticreg\\datasets\\affairs.csv")

affair= affair.drop(["Unnamed: 0"],axis=1)
affair["gender"].value_counts()
affair["age"].value_counts()
affair["yearsmarried"].value_counts()
affair["religiousness"].value_counts()
affair["education"].value_counts()
affair["occupation"].value_counts()
affair["rating"].value_counts()

plt.boxplot(affair["age"])
plt.boxplot(affair["yearsmarried"])
plt.boxplot(affair["education"])
plt.boxplot(affair["occupation"])
plt.boxplot(affair["rating"])
plt.boxplot(affair["religiousness"])
plt.boxplot(affair["affairs"])

#calculating the interquantile range
q25 = affair.quantile(0.25)
q75 = affair.quantile(0.75)
iqr= q75-q25
lower_bound = q25 - (1.5 * iqr)
upper_bound = q75 + (1.5* iqr)

out_25= (affair < (q25 - (1.5 * iqr))) 
out_75 = (affair> (q75+(1.5 * iqr)))

#we now have the IQR scores, it’s time to get hold on outliers.
#The above code will give an output with some true and false values. 
#The data point where we have False that means these values are valid whereas True indicates presence of an outlier.

##Checking for outliers below lower_bound.
out_25["age"].value_counts()
out_25["affairs"].value_counts()
out_25["children"].value_counts()
out_25["education"].value_counts()
out_25["gender"].value_counts()
out_25["occupation"].value_counts()
out_25["rating"].value_counts()
out_25["religiousness"].value_counts()
out_25["yearsmarried"].value_counts()
## There are no outliers below lower bound.


## Checking for outliers above upper bound
out_75["age"].value_counts() ## 22 are outliers
out_75["affairs"].value_counts() ## 150 are outliers
out_75["children"].value_counts()
out_75["education"].value_counts()
out_75["gender"].value_counts()
out_75["occupation"].value_counts()
out_75["rating"].value_counts()
out_75["religiousness"].value_counts()
out_75["yearsmarried"].value_counts()

## log transformation for age, to convert the outliers
x= np.log(affair["age"])
plt.boxplot(x)
q_25 = np.log(affair["age"]).quantile(0.25)
q_75 = np.log(affair["age"]).quantile(0.75)
iqr2 = q_75-q_25
lower_bound_one = q_25 - (1.5 * iqr2)
upper_bound_one = q_75 + (1.5* iqr2)

out_25_one= (np.log(affair["age"]) < lower_bound_one) 
out_75_one = (np.log(affair["age"])> upper_bound_one)

out_25_one.value_counts()## no outliers
out_75_one.value_counts()## no outliers

#we now have the IQR scores, it’s time to get hold on outliers.
#The above code will give an output with some true and false values. 
#The data point where we have False that means these values are valid whereas True indicates presence of an outlier.

###
affair.isnull().sum()

## Creating dummies for gender and children
affairs_dummy = pd.get_dummies(affair[["gender","children"]])
#affair2 =pd.get_dummies(affair,drop_first =True)

affair1= affair.drop(["gender"], axis=1)
affair2 = affair1.drop(["children"],axis=1)


affair2= pd.concat([affair2,affairs_dummy],axis=1)

affair2["AF"]=1
affair2.loc[affair2.affairs==0,"AF"]=0
affair2=affair2.drop(["affairs"],axis=1)
affair2=affair2.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]

affair2.info()
## coverting religiousness and rating into factors, as it is ordinal data

affair2["religiousness"]= pd.Categorical(affair2["religiousness"])
affair2["rating"]=pd.Categorical(affair2["rating"])
#affair2["occupation"]=pd.Categorical(affair2["occupation"])
#affair2["gender_male"]=pd.Categorical(affair2["gender_male"])
#affair2["children_yes"]= pd.Categorical(affair2["children_yes"])


#creating the model
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(affair2,test_size = 0.2)

train_data.to_csv("training_data.csv",encoding="utf-8")
test_data.to_csv("testing_data.csv",encoding= "utf-8")
import statsmodels.formula.api as sm
## trailone
model_one = sm.logit("AF~np.log(age)+yearsmarried+religiousness+rating+occupation+education+gender_female+gender_male+children_no+children_yes", data= train_data).fit()
model_one.summary()
model_one.summary2()
## AIC= 486

from scipy import stats
import scipy.stats as st
st.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)

y_pred = model_one.predict(train_data)
train_data["pred_prob"]=y_pred

train_data["Af_val"]=np.zeros(480)
train_data.loc[y_pred>=0.50,"Af_val"]=1
train_data.Af_val

from sklearn.metrics import classification_report
classification = classification_report(train_data["Af_val"],train_data["AF"])

#confusion matrix
confusion_matrx = pd.crosstab(train_data["AF"],train_data["Af_val"])

##accuracy
accuracy = (345+34)/480 ## 78.9%

##ROC curve
from sklearn import metrics
#fpr=> false positive rate
#tpr=> true positive rate
fpr,tpr,threshold = metrics.roc_curve(train_data["AF"], y_pred)

plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc = metrics.auc(fpr,tpr) ## 75.57%

## test data

test_pred = model_one.predict(test_data)

test_data["pred_test"] = test_pred

test_data["test_val"]= np.zeros(121)

test_data.loc[test_pred>=0.5,"test_val"]=1

## confusion matrix
confusion_matrix_test1 = pd.crosstab(test_data["AF"],test_data["test_val"])

##accuracy
accuracy_test= (81+5)/121 ## 71.03%

############ trails for perfect cut-off value 0.58

y_pred1 = model_one.predict(train_data)
train_data["pred_prob1"]=y_pred1

train_data["Af_val1"]=np.zeros(480)
train_data.loc[y_pred1>=0.58,"Af_val1"]=1
train_data.Af_val1

train_data["Af_val1"].value_counts()

from sklearn.metrics import classification_report
classification = classification_report(train_data["Af_val1"],train_data["AF"])

#confusion matrix
confusion_matrx1= pd.crosstab(train_data["AF"],train_data["Af_val1"])

##accuracy
accuracy1 = (352+19)/480 ## 77.29

fpr2,tpr2,threshold2 = metrics.roc_curve(train_data["AF"], y_pred1)

plt.plot(fpr2,tpr2);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc1 = metrics.auc(fpr, tpr) ##75.575
## test data

test_pred1 = model_one.predict(test_data)

test_data["pred_test1"] = test_pred1

test_data["test_val1"]= np.zeros(121)

test_data.loc[test_pred>=0.58,"test_val1"]=1

## confusion matrix
confusion_matrix_test2 = pd.crosstab(test_data["AF"],test_data["test_val1"])

##accuracy
accuracy_test1= (85+3)/121 ## 72.727%

########################################################trail two##############################################
### as all the variables are insignificant i would remove education, occupation, gender and children to build the model.

model_two = sm.logit("AF~np.log(age)+yearsmarried+religiousness+rating", data = train_data).fit()
model_two.summary()
model_two.summary2()
## As rating and religiousness is ordinal data, we converted it into factor using pd.Categorical.
## Few of the factors are still insignificant. Consodering them and building the model because they are important for the model. 
## AIC=484

y_pred2 = model_two.predict(train_data)
train_data["pred_prob2"]=y_pred2
train_data["Af_val2"]=np.zeros(480)
train_data.loc[y_pred2>=0.5,"Af_val2"]=1

classification1 = classification_report(train_data["Af_val2"],train_data["AF"])

## confusion matrix
confusion_matrix2 = pd.crosstab(train_data["AF"], train_data["Af_val2"])

## accuracy
accuracy2 = (347+26)/480
##77.7

fpr3,tpr3, threshold3 = metrics.roc_curve(train_data["AF"],y_pred2)

plt.plot(fpr,tpr);plt.xlabel("false positive rate");plt.ylabel("True positive rate")

roc_auc2 = metrics.auc(fpr,tpr)##75.575

## for test data
test_pred2= model_two.predict(test_data)
test_data["pred_test2"]=test_pred2
test_data["test_val2"]= np.zeros(121)
test_data.loc[test_pred2>=0.5,"test_val2"]=1

confusion_matrix_test3 = pd.crosstab(test_data["AF"],test_data["test_val2"])

accuracy2 = (78+6)/121 ## 69.4%


## Checking for different cut off values 0.58

y_pred3 = model_two.predict(train_data)

train_data["pred_prob3"]=y_pred3
train_data["Af_val3"]= np.zeros(480)

train_data.loc[y_pred3>=0.54,"Af_val3"]=1

classification2 = classification_report(train_data["Af_val3"],train_data["AF"])

# confusion matrix
confusion_matrix3 = pd.crosstab(train_data["AF"],train_data["Af_val3"]) 

accuracy3= (351+22)/480
## 77.7

fpr4,tpr4,threshold4 = metrics.roc_curve(train_data["AF"],y_pred3)

plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc3 = metrics.auc(fpr,tpr) ## 75.575

## test data

test_pred3 = model_two.predict(test_data)
test_data["pred_test3"]=test_pred3
test_data["test_val3"] = np.zeros(121)
test_data.loc[test_pred3>=0.54,"test_val3"]=1

confusion_matrix_test4= pd.crosstab(test_data["AF"],test_data["test_val3"])
accuracy_test3 = (85+5)/121 ##74.3

###model_two is selected and with the cut off value as 0.54. As it has least fpr and high tpr.