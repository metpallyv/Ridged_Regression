# Implementation of Ridged regression

__author__ = 'Vardhaman'
import sys
import csv
import math
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import inv
from numpy import *#genfromtxt
feature = []

#load the input file
def load_csv(file):
    X = genfromtxt(file, delimiter=",",dtype=str)
    #print(X)
    return (X)

def random_numpy_array(ar):
    np.random.shuffle(ar)
    #print(arr)
    arr = ar
    #print(arr)
    return arr

def centered_X(matrix,me):
    with np.errstate(divide='ignore'):
        a = matrix
        mean_list = []
        if me == 0:
            b = np.apply_along_axis(lambda x: (x-np.mean(x)),0,a)
            tmp = a.shape[1]
            for i in range(tmp):
                mean_list.append(np.mean(a[:,i]))
            return b,mean_list

        else:
            res = np.empty(shape=[a.shape[0],0])
            for i in range(a.shape[1]):
                col = matrix[:, i]
                mean_val = me[i]
                b = np.apply_along_axis(lambda x: (x-mean_val),0,col)
                #print b.shape,res.shape
                #res = np.concatenate((res, b), axis=1)
                res = np.insert(res,res.shape[1],values=b.flatten(),axis=1)
                #res = np.insert()
        #res = np.nan_to_num(res)
    return res,me

def centered_Y(Y):
    b = np.apply_along_axis(lambda x: (x-np.mean(x)),0,Y)
    return b

def generate_set(X,poly):
    X = np.array(X,dtype=float)
    Y = X[:,-1]
    X = X[:,:-1]
    X_initial = X[:,:-1]
    a = range(2,poly+1)
    for i in a:
        b = np.power(X_initial,i)
        X = np.append(X,b,axis=1)
    X = np.insert(X,X.shape[1],values=Y.flatten(),axis=1)
    num_test = round(0.1*(X.shape[0]))
    start = 0
    end = num_test
    test_attri_list =[]
    test_class_names_list =[]
    training_attri_list = []
    training_class_names_list = []
    for i in range(10):
        X_test = X[start:end , :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        y_training = X_training[:, -1]
        y_test = X_test[:, -1]
        X_training = X_training[:,:-1]
        X_test = X_test[:,:-1]
        X_training_normalized,me = centered_X(X_training,0)
        X_test_normalized,me = centered_X(X_test,me)
        #print X_training_normalized.shape,X_test_normalized.shape
        #y_test = centered_Y(y_test)
        y_test = y_test.flatten()
        #y_training = centered_Y(y_training)
        y_training = y_training.flatten()
        test_attri_list.append(X_test_normalized)
        test_class_names_list.append(y_test)
        training_attri_list.append(X_training_normalized)
        training_class_names_list.append(y_training)
        start = end
        end = end+num_test

    return test_attri_list,test_class_names_list,training_attri_list,training_class_names_list

def normal_equation(x,y,lam):
    # calculate weight vector with the formula inverse of(x.T* x)*x.T*y
    a = dot(x.transpose(),x)
    #z = inv(dot(x.transpose(), x)+ lam*np.identity(x.shape[0]))
    z = inv(a+lam*np.identity(a.shape[0]))
    theta = dot(dot(z, x.transpose()), y)

    return theta

def compute_rmse_sse(x,y,theta):
    m = y.size
    #y = map(lambda x: x+np.mean(x),y)
    pred = x.dot(theta) + np.mean(y)
    #print(pred)
    error = pred - y
    sse = error.T.dot(error)/float(m)
    rmse = math.sqrt(sse)
    #print"SSE:",sse,"RMSE:",rmse
    return rmse,sse

if __name__ == "__main__":
    if len(sys.argv) == 3:
        newfile = sys.argv[1]
        poly = int(sys.argv[2])
        num_arr = load_csv(newfile)
        num_arr = random_numpy_array(num_arr)
        #Divide the data into 10 cross training and test data
        test_x,test_y,training_x,training_y = generate_set(num_arr,poly)
        #Apply normal form equation for all 10 cross data

        a = np.arange(0,10.2,0.2)
        #print(a)
        trainingRMSEValues = []
        testRMSEValues = []

        for l in a:
            rmse_training = []
            rmse_test = []
            for i in range(10):

                theta = normal_equation(training_x[i],training_y[i],l)
                #calculate the rmse and sse for each fold
                rmse1,see2 = compute_rmse_sse(training_x[i],training_y[i],theta)
                rmse2,sse1 = compute_rmse_sse(test_x[i],test_y[i],theta)
                rmse_training.append(rmse1)
                rmse_test.append(rmse2)
                #print "RMSE for training",rmse1
                #print "RMSE for test",rmse2

            meanTrainingRMSE = sum(rmse_training)/float(10)
            trainingRMSEValues.append(meanTrainingRMSE)
            meanTestingRMSE = sum(rmse_test)/float(10)
            testRMSEValues.append(meanTestingRMSE)

            print "Average RMSE for Training for lamba:",l,"is",sum(rmse_training)/float(len(rmse_training))
            print "Average RMSE for Test for lamba: ",l,"is",sum(rmse_test)/float(len(rmse_test))

        plt.suptitle("Ridged regression Data plot")
        plt.plot(a,trainingRMSEValues)
        plt.ylabel("traning RMSE")
        plt.plot(a,testRMSEValues)
        plt.ylabel("RMSE")
        plt.xlabel("Lambda ")
        fileName = "RidgedRegression";
        plt.savefig(fileName)
