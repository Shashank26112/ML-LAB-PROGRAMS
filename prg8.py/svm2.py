import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

digits=datasets.load_digits()
clf=svm.SVC(gamma=0.0001,C=100)
X,Y=digits.data[:-10],digits.target[:-10]
clf.fit(X,Y)
print(clf.predict(digits.data[:-10]))
plt.imshow(digits.images[0],interpolation='nearest')
plt.show()