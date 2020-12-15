# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:06:58 2020

@author: Vishal
"""
#####################importing###################

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import LinearSVC

from plotly.offline import plot,iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.utils import shuffle



df = pd.read_csv("creditcard.csv")
df.head(n=5)

df.hist()
plt.show()
df.plot(kind='density', subplots=True, layout=(6,6), sharex=False)
plt.show()


fraud_data = df[df['Class']==1]
genuine_data = df[df['Class']==0]
print(df['Class'].value_counts())

fig=px.bar(x = ["Genuine","Fraud"],y = [len(genuine_data),len(fraud_data)],labels={'x':'Type of Transactions','y':'Number of Transactions'},title="Types of Transaction by Number")
fig.show()


##for Fraud Transactions
fig=px.scatter(x = fraud_data.Time,y = fraud_data.Amount,labels={'x':'Amount','y':'Time'},title="Amount vs Time for Fraud Transactions" )
fig.show()

##for Genuine Transactions
fig=px.scatter(x = genuine_data.Time,y = genuine_data.Amount,labels={'x':'Amount','y':'Time'},title="Amount vs Time for Genuine Transactions")
fig.show()


###################### feature scaling & SMOTE #################
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df["Amount"]=scaler.fit_transform(np.array(df["Amount"]).reshape(-1,1))

import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
x = df.drop("Class", axis=1)
y = df['Class']

x=x.values
y=y.values
#smote resampling

from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_resample(x, y)

#clf_smote = LinearSVC().fit(X_resampled, y_resampled)


X_resampled, y_resampled = shuffle(X_resampled, y_resampled)
#dividing the data
x_train,x_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.3, random_state = 0)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state = 0)



###### Let us use Logistic Regression to Classify our data ############
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy: {}%".format(round(metrics.accuracy_score(y_test, y_pred)*100)))
print("Recall: {}%".format(round(metrics.recall_score(y_test, y_pred)*100)))
print("Precision: {}%".format(round(metrics.precision_score(y_test, y_pred)*100)))
print("F1: {}%".format(round(metrics.f1_score(y_test, y_pred)*100)))    
m_confusion_test = metrics.confusion_matrix(y_test, y_pred)
pd.DataFrame(data = m_confusion_test, columns = ['Predicted Genuine', 'Predicted Fraud'],index = ['Actual Genuine', 'Actual Fraud'])



###### AUC #############

from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()





############ Random FOrest now ##################
from sklearn.ensemble import RandomForestClassifier
#clf=RandomForestClassifier(n_estimators=7, max_depth=2)
clf=RandomForestClassifier()
#Train the model
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

# performance evaluation
print("Accuracy: {}%".format((metrics.accuracy_score(y_test, y_pred)*100)))
print("Recall: {}%".format((metrics.recall_score(y_test, y_pred)*100)))
print("Precision: {}%".format((metrics.precision_score(y_test, y_pred)*100)))
print("F1: {}%".format((metrics.f1_score(y_test, y_pred)*100)))    
m_confusion_test = metrics.confusion_matrix(y_test, y_pred)
pd.DataFrame(data = m_confusion_test, columns = ['Predicted Genuine', 'Predicted Fraud'],index = ['Actual Genuine', 'Actual Fraud'])
 
    
 
########### K FOld cross validation ############## 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
kfold = KFold(n_splits=10, random_state=0)
cv_result = cross_val_score(estimator=clf,X=x_train,y=y_train, cv = kfold,scoring = "accuracy")
print("Accuracy: {}%".format(round(cv_result.mean()*100)))
cv_result = cross_val_score(estimator=clf,X=x_train,y=y_train, cv = kfold,scoring = "precision")
print("Precision: {}%".format(round(cv_result.mean()*100)))
cv_result = cross_val_score(estimator=clf,X=x_train,y=y_train, cv = kfold,scoring = "recall")
print("Recall: {}%".format(round(cv_result.mean()*100)))


############# Finally ANN ####################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x_test=scaler.fit_transform(x_test)
x_train=scaler.fit_transform(x_train)


model=Sequential()
model.add(Dense(10, activation = "relu", input_shape = (30,)))
model.add(Dense(20, activation = "relu"))
model.add(Dense(40, activation = "relu"))
model.add(Dense(60, activation = "relu"))
model.add(Dense(80, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
 
model.compile(loss = "binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

#actual execution:
history = model.fit(np.array(x_train), np.array(y_train), validation_split =0.2, epochs = 200, batch_size = x_train.shape[0])





################ ann accuracy graph ##########
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()







#Boxplot
history.history
loss = history.history["loss"]
val_loss = history.history["val_loss"]
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
tva =pd.DataFrame({
    "TrainingAcc":accuracy,
    "Val_Acc": val_accuracy
})
tva.boxplot()

y_pred = np.round(model.predict(np.array(x_test)))

#results

print("Accuracy: {}%".format(round(metrics.accuracy_score(y_test, y_pred)*100)))
print("Recall: {}%".format(round(metrics.recall_score(y_test, y_pred)*100)))
print("Precision: {}%".format(round(metrics.precision_score(y_test, y_pred)*100)))
print("F1: {}%".format(round(metrics.f1_score(y_test, y_pred)*100)))

#######saving the model #########
from _pickle import dump
filename = 'Final_Model.sav'
dump(clf, open(filename, 'wb'))

model.save("fraud_model_smote.h5")



###############  SVM #####################
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


######### New Plot methods ##########


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score



# Compute ROC curve and ROC area for each class

n_classes = y.shape[0]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,170589,100):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.plot(list(fpr.keys()), list(tpr.values()),list(roc_auc.values()))


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()











plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()










