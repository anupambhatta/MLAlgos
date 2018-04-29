# -*- coding: utf-8 -*-
"""
This file uses the classes from mllib and shows some basic examples.
"""

#Import the DecisionTreeClassifier
from mllib.DecisionTreeClf import DecisionTreeClf

from sklearn.datasets import load_iris
X_ = load_iris().data[:,2:]
y_ = load_iris().target

#Create object of the DecisionTree class
mytree = DecisionTreeClf(2)
#Train the tree
mytree.fit(X_, y_)
#print the tree
mytree.printTreePreOrder()
#test the model
y_pred = mytree.predict([[5, 1.5]])
print(y_pred)