import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

#load the dataset
cancer_data=datasets.load_breast_cancer()
#print(cancer_data)
print(cancer_data['target'])

#split the dataset
X_train,X_test,Y_train,Y_test=train_test_split(cancer_data.data,cancer_data.target,test_size=0.40,random_state=30)

#generate the model
cls=svm.SVC(kernel="linear")

#tarin the model
cls.fit(X_train,Y_train)

#PREDICT THE RESPONSE
pred=cls.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test,y_pred=pred))
print("Precision Score:", metrics.precision_score(Y_test,y_pred=pred))
print("Recall:",metrics.recall_score(Y_test,y_pred=pred))
print(metrics.classification_report(Y_test,y_pred=pred))
