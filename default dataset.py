# -*- coding: utf-8 -*-

from pandas import read_csv
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

# define the dataset location

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'

# load the csv file as a data frame

df = read_csv(url, header=None)
data = df.values

# split into input and output elements

X, y = data[:, :-1], data[:, -1]

# label encode the target variable
y = LabelEncoder().fit_transform(y)

# summarize distribution

counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    
# plot the distribution

pyplot.bar(counter.keys(), counter.values())
pyplot.show()


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#We use Support Vector classifier as a classifier
from sklearn.svm import SVC

#training the classifier using X_Train and y_train 
clf = SVC(kernel = 'linear').fit(X_train,y_train)
clf.predict(X_train)

#Testing the model using X_test and storing the output in y_pred
y_pred = clf.predict(X_test)

# Build confusion matrix from ground truth labels and model predictions
conf_mat = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', conf_mat)
# Plot matrix
plt.matshow(conf_mat)
plt.colorbar()
plt.ylabel('Real Class')
plt.xlabel('Predicted Class')
plt.show()