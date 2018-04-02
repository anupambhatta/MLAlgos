# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:20:14 2018

My Implementation for the Decision Tree Classifier
The code can be used to simple classification based on Decision Trees.
It uses the CART algorithm to train the Decision Trees.

Limitations:
It takes only one hyperparamater which is the max depth of the tree

@author: Anupam Bhattacharjee
"""

import numpy as np

class DecisionTreeClf:
    recursion_count = 0 #static variable to maintain recursion count
    classes = [] #static variable to maintain the classes
    
    def __init__(self, max_depth=100):
        '''
        Initilize an object of DecisionTreeClf
        '''
        self.max_depth = max_depth
        self.decision_var = None
        self.decision_val = None
        self.gini = None
        self.samples = None
        self.values = []
        self.left = None
        self.right = None
        
    def __repr__(self):
        '''
        String representation of the DecisionTreeClf object
        '''
        return "-----------\nX_{} <= {}\ngini = {} \nsamples = {}\nvalues = {}\
        \nleft = {}\nright = {}\n------------\n".format(self.decision_var, \
        self.decision_val, self.gini, self.samples, self.values, \
        'Y' if self.left else 'N' , 'Y' if self.right else 'N')

    def getClasses(y):
        '''
        Returns array of unique values (classes) given an array y
        '''
        return np.unique(y)
    
    def getGini(self, values, sample_size):
        '''
        Calculates the gini index given the values of each class type 
        and sample size
        '''
        gini = 1
        for i in range(len(values)):
            gini = gini - ((values[i]/sample_size)**2)
        return gini
    
    def costFunction(self, mLeft, mRight, m, gLeft, gRight):
        '''
        Calculates the cost function given the mleft, mright, m, gleft and gright
        where, mleft = number of instances in left subset
               mright = number of instances in right subset
               m = total number of instances
               gleft = gini index of left side
               gright = gini index of right side
        '''
        return (((mLeft/m)*gLeft) + ((mRight/m)*gRight))
    
    def getCountsPerClass(self, y):
        '''
        Returns an array with counts for each class type
        '''
        values = np.zeros(len(DecisionTreeClf.classes), dtype=int)
        for i in range(len(y)):
            for j in range(len(DecisionTreeClf.classes)):
                if y[i] == DecisionTreeClf.classes[j]:
                    values[j] += 1
                    break
        return values
    
    def getMandG(self, X, y, tk, k):
        '''
        Returns mleft, mright, m, gleft and gright
        given a variable k and a threshold value tk
        '''
        m = len(y)
        mleft = 0
        mright = 0
        values_left = np.zeros(len(DecisionTreeClf.classes), dtype=int)
        values_right = np.zeros(len(DecisionTreeClf.classes), dtype=int)
        for i in range(len(X)):
            if X[i][k] <= tk:
                mleft += 1
                for j in range(len(DecisionTreeClf.classes)):
                    if y[i] == DecisionTreeClf.classes[j]:
                        values_left[j] += 1
                        break
            else:
                mright += 1
                for j in range(len(DecisionTreeClf.classes)):
                    if y[i] == DecisionTreeClf.classes[j]:
                        values_right[j] += 1
                        break
        return mleft, mright, m, self.getGini(values_left, mleft), self.getGini(values_right, mright)
    
    def minimizeCost(self, X, y):
        '''
        Loops through each variable k and each value of k
        to find the value tk that produces minimum cost and 
        returns the k and tk corresponding to the minimum cost.
        '''
        min_cost = 1
        decision_var = 0
        decision_val = X[0][0]
        prev = X[0][0]
        for i in range(len(X)):
            for k in range(len(X[i])):
                if (i !=0 or k!=0) and prev == X[i][k] :
                    continue
                prev = X[i][k]
                #get the all m and g for a tk
                mLeft, mRight, m, gLeft, gRight = self.getMandG(X, y, X[i][k], k)
                curr_cost = self.costFunction(mLeft, mRight, m, gLeft, gRight)
                if(min_cost > curr_cost):
                    min_cost = curr_cost
                    decision_var = k
                    decision_val = X[i][k]
        return decision_var, decision_val
    
    def splitLeftAndRight(self, X, y, dec_var, dec_val):
        '''
        Splits the X and y into left and right subsets given a decision variable
        and value.
        '''
        X_left = [] 
        Y_left = [] 
        X_right = [] 
        Y_right = []
        for i in range(len(X)):
            if X[i][dec_var] <= dec_val:
                X_left.append(X[i])
                Y_left.append(y[i])
            else:
                X_right.append(X[i])
                Y_right.append(y[i])
        return np.asarray(X_left), np.asarray(Y_left), np.asarray(X_right), np.asarray(Y_right)
    
    def fit(self, X, y):
        '''
        Recursively creates the decision tree given X and y
        '''
        #during the first call create the classes 
        if DecisionTreeClf.recursion_count == 0: 
            DecisionTreeClf.classes = DecisionTreeClf.getClasses(y)
        self.values = self.getCountsPerClass(y)
        self.samples = len(y)
        self.gini = self.getGini(self.values, self.samples)
        #create a sub-tree only if gini index is more than 0 and the max-depth has not been reached 
        if self.gini > 0 and self.max_depth > 0:
            DecisionTreeClf.recursion_count += 1
            #decision based on min cost
            self.decision_var, self.decision_val = self.minimizeCost(X, y)
            #Now split the data into Xleft, Yleft and Xright, Yright
            Xleft, Yleft, Xright, Yright = self.splitLeftAndRight(X, y,\
                                      self.decision_var, self.decision_val)
            #now create the left and rifht sub-trees and for the corresponding data
            if len(Xleft) > 0:
                self.left = DecisionTreeClf(self.max_depth-1)
                self.left.fit(Xleft, Yleft)
            if len(Xright) > 0:
                self.right = DecisionTreeClf(self.max_depth-1)
                self.right.fit(Xright, Yright)

    def printTreeInOrder(self):
        '''
        Prints the tree in order
        '''
        if self is not None:
            print(self)
        if self.left is not None:
            self.left.printTreeInOrder()
        if self.right is not None:
            self.right.printTreeInOrder()
   
    def getMaxClass(self):
        '''
        Returns the class based on the max prbability
        '''
        max_ = 0.0
        max_ind = 0
        for i in range(len(self.values)):
            if max_ < self.values[i]:
                max_ = self.values[i]
                max_ind = i
        return max_ind
        
    def predictClass(self, X_pred_):
        '''
        Finds the leaf node given the independent variables
        and returns the class based on the highest probability
        '''
        temp = self
        while temp.decision_var is not None :
            if X_pred_[temp.decision_var] <= temp.decision_val:
                temp = temp.left
            else:
                temp = temp.right
        return temp.getMaxClass()
        
    def predict(self, X_pred):
        '''
        Predicts the class given an array of independent variables
        '''
        predictions = []
        for i in range(len(X_pred)):
            predictions.append(self.predictClass(X_pred[i]))
        return predictions