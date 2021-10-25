"""
Created on Sat Jan  4 19:55:31 2020

@author: merve
"""

import numpy as np
import pandas as pd
import time
tic = time.time()
#import the data and the labels
df = pd.read_csv("hw07_target1_training_data.csv")
df2 = pd.read_csv("hw07_target2_training_data.csv")
df3 = pd.read_csv("hw07_target3_training_data.csv")

#drop categorical features
df= df.drop(columns=["VAR45","VAR47","VAR75"])
df2= df2.drop(columns=["VAR32","VAR65","VAR195"])
df3= df3.drop(columns=["VAR36","VAR153"])

X_test=pd.read_csv("hw07_target1_test_data.csv")
X_test2=pd.read_csv("hw07_target2_test_data.csv")
X_test3=pd.read_csv("hw07_target3_test_data.csv")
#drop categorical features
X_test= X_test.drop(columns=["VAR45","VAR47","VAR75"])
X_test2= X_test2.drop(columns=["VAR32","VAR65","VAR195"])
X_test3= X_test3.drop(columns=["VAR36","VAR153"])
#fill not-a-numbers with the median
X_test.fillna(X_test.median(), inplace=True)
X_test2.fillna(X_test2.median(), inplace=True)
X_test3.fillna(X_test3.median(), inplace=True)



#fill not-a-numbers with the median
df.fillna(df.median(), inplace=True)
df2.fillna(df2.median(), inplace=True)
df3.fillna(df3.median(), inplace=True)

#import labels
Labels = pd.read_csv("hw07_target1_training_label.csv")
Labels2 = pd.read_csv("hw07_target2_training_label.csv")
Labels3 = pd.read_csv("hw07_target3_training_label.csv")

Y=np.array(Labels)[:,1]
Y2=np.array(Labels2)[:,1]
Y3=np.array(Labels3)[:,1]
X=df
X2=df2
X3=df3

# train a random forest
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=80,max_depth=7)
rfc2=RandomForestClassifier(n_estimators=80,max_depth=5)
rfc3=RandomForestClassifier(n_estimators=80,max_depth=7)
#grid search
#param_grid = {'n_estimators': [80,100], 'max_depth': list(np.arange(5,8))}
#rfc = GridSearchCV(rfc, param_grid,scoring=('roc_auc'), cv=10)
#fit and evaluate posterior for training set
Y_rf = rfc.fit(X, Y)
Y_pred=rfc.predict_proba(X)[:,1]
Y_rf2 = rfc2.fit(X2, Y2)
Y_pred2=rfc2.predict_proba(X2)[:,1]
Y_rf3 = rfc3.fit(X3, Y3)
Y_pred3=rfc3.predict_proba(X3)[:,1]


#rfc.best_params_
#calculate mean auroc using 10-fold cross validation
from sklearn.model_selection import cross_validate
scores = cross_validate(rfc, X, Y, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 1 is:")
print(np.mean(scores['test_score']))
scores2 = cross_validate(rfc2, X2, Y2, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 2 is:")
print(np.mean(scores2['test_score']))
scores3 = cross_validate(rfc3, X3, Y3, cv=10, scoring=('roc_auc'), return_train_score=True)
print("Mean AUROC for target 3 is:")
print(np.mean(scores3['test_score']))

# evaluate posteriors on the test set
Y_test=Y_rf.predict_proba(X_test)
#concatanate with ID
Y_o=np.concatenate((np.array(X_test["ID"]).reshape((-1,1)),Y_test[:,1].reshape((-1,1))),1)
Y_test2=Y_rf2.predict_proba(X_test2)
#concatanate with ID
Y_o2=np.concatenate((np.array(X_test2["ID"]).reshape((-1,1)),Y_test2[:,1].reshape((-1,1))),1)
Y_test3=Y_rf3.predict_proba(X_test3)
#concatanate with ID
Y_o3=np.concatenate((np.array(X_test3["ID"]).reshape((-1,1)),Y_test3[:,1].reshape((-1,1))),1)

# write predictions to csv files
pd.DataFrame(Y_o).to_csv("hw07_target1_test_predictions.csv",header=["ID","TARGET"],index=None)
pd.DataFrame(Y_o2).to_csv("hw07_target2_test_predictions.csv",header=["ID","TARGET"],index=None)
pd.DataFrame(Y_o3).to_csv("hw07_target3_test_predictions.csv",header=["ID","TARGET"],index=None)

# plot ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(Y, Y_pred)
fpr2, tpr2, threshold = metrics.roc_curve(Y2, Y_pred2)
fpr3, tpr3, threshold = metrics.roc_curve(Y3, Y_pred3)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic for target 1,2,3')
plt.plot(fpr, tpr, 'b')
plt.plot(fpr2, tpr2, 'r')
plt.plot(fpr3, tpr3, 'g')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(('target 1','target 2','target 3'))
plt.show()
