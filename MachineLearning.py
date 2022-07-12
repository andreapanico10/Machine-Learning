#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
import seaborn as sns

def costLoopBased(x,y, theta):
    #theta= np.zeros((x.shape[1],1))
    m = x.shape[0]
    n = x.shape[1]
    J = 0
    
    elapsed_time = 0
    elapsed_time -= time.time()
    for i in range(m): # iterate on over all the input example
        h = 0
        for j in range (n): # iterate over each input feature
            
            h += theta[j]*x[i][j]
            
        J += (h - y[i])**2 ##individual loss
    J = J/(2*m),3
    elapsed_time += time.time()
    elapsed_time *= 1000 # porto i secondi in millisecondi
    
    return np.round(J[0],3), np.round(elapsed_time,3)


def costVectorial(x,y, theta):
    #theta= np.zeros((x.shape[1],1))
    m = x.shape[0]

    J = 0
    
    elapsed_time = 0
    elapsed_time -= time.time()

    J = (x.dot(theta) - y).T.dot((x.dot(theta) - y))
    J = J/(2*m)

    elapsed_time += time.time()
    elapsed_time *= 1000 # porto i secondi in millisecondi
    
    return np.round(J[0],3), np.round(elapsed_time,3)

def costNumpy(x,y, theta):
    #theta= np.zeros((x.shape[1],1))
    m = x.shape[0]

    J = 0
    
    elapsed_time = 0
    elapsed_time -= time.time()

    # strategia ibrida
    # in ordine
    
    # vettore colonna x*theta - vettore colonna y -> ottengo vettore colonna [97 x 1]
    hyphotesys = x.dot(theta) 
    # faccio il quadrato, sempre [97 x 1]
    square = np.square(hyphotesys - y)
    # sommo tutti e 97 gli elementi -> ottengo scalare [1 x 1]
    J = np.sum(square)
    J = J/(2*m)
    
    elapsed_time += time.time()
    elapsed_time *= 1000 # porto i secondi in millisecondi
    
    return np.round(J,3), np.round(elapsed_time,3)

def gradientDescentLoop(x, y, theta, alpha = 0.003, max_iters = 20000, early = False, epsilon = 0.00001):
    # theta = np.zeros((x.shape[1],1))
    m = x.shape[0]
    n = x.shape[1]
    J_history = np.zeros((max_iters,1))
    theta_history = []

    theta_gd = np.copy(theta)
    
    elapsed_time = 0
    elapsed_time -= time.time()

    for iter in range (max_iters):
        J_history[iter],_ = costLoopBased(x,y, theta_gd)
        theta_history.append(np.copy(theta_gd))
        h = np.zeros((m,1))
        partial = np.zeros((n,1))
        
        for i in range(m):
            for j in range(n):
                h[i] += theta_gd[j]*x[i][j]
            
            for j in range(n):
                partial[j] += (h[i] - y[i])*x[i][j]
                
        #simultaneously update each parameter        
        for j in range(n):
            theta_gd[j] = theta_gd[j] - (alpha/m)*partial[j]
        if(early):
            if(iter != 0):
                if(np.abs(J_history[iter] - J_history[iter-1]) < epsilon):
                        theta_history.append(theta_gd)
                        elapsed_time += time.time()
                        elapsed_time *= 1000 # porto i secondi in millisecondi
                        return theta_gd, J_history[J_history != 0], round(elapsed_time,3), theta_history
    
    theta_history.append(np.copy(theta_gd))
    elapsed_time += time.time()
    elapsed_time *= 1000 # porto i secondi in millisecondi

    return theta_gd, J_history, round(elapsed_time,3), theta_history
    

def gradientDescentVectorized(x, y, theta, alpha = 0.003, max_iters = 100000, early = False, epsilon = 0.00001):
    # theta = np.zeros((x.shape[1],1))
    m = x.shape[0]

    J_history = np.zeros((max_iters,1))
    theta_history = []

    theta_gd = np.copy(theta)
    
    elapsed_time = 0
    elapsed_time -= time.time()

    for iter in range (max_iters):
        J_history[iter],_ = costVectorial(x,y, theta = theta_gd)
        
        theta_gd = theta_gd - (alpha/m) *x.T.dot(x.dot(theta_gd) - y)
        theta_history.append(np.copy(theta_gd))        
        
        if(early):
            if(iter != 0):
                if(np.abs(J_history[iter] - J_history[iter-1]) < epsilon):
                        theta_history.append(theta_gd)
                        elapsed_time += time.time()
                        elapsed_time *= 1000 # porto i secondi in millisecondi
                        return theta_gd, J_history[J_history != 0], round(elapsed_time,3), theta_history
    
    theta_history.append(np.copy(theta_gd))
    elapsed_time += time.time()
    elapsed_time *= 1000 # porto i secondi in millisecondi

    return theta_gd, J_history, round(elapsed_time,3), theta_history

def scatter_plot_dati(x,y, label='label', xlabel = 'xlabel', ylabel = 'ylabel'):
    plt.figure(figsize = (12,5)) # dichiaro l'ambiente per disegnare
    # (12,5) vuole prima colonne e poi righe
    plt.scatter(x,y, marker='x', c='red', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.1)
    plt.legend()
    plt.show()
    
def cost_history_plot(cost_history, label='', xlabel = 'iterations', ylabel = 'ylabel'):
    plt.figure(figsize = (12,8))
    plt.plot(np.arange(cost_history.shape[0])+1, cost_history, label = label);
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
    #prende in input la x con la colonna di 1 già aggiunta 
def plot_linear_hypothesis_with_data(x,y, theta, xlabel = 'xlabel', ylabel = 'ylabel'):
    plt.figure(figsize=(10,5))
    plt.scatter(x[:,1], y, marker = 'x', c='r')
    plt.plot(x[:,1], x.dot(theta), label = '{} + {}*x'.format(theta[0], theta[1]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc = 'upper left', ncol= 4)
    plt.show()
    
    #x con la colonna di 1 già aggiunta
def countour_plot(x,y):
    B0 = np.linspace(-80,80, 50)
    B1 = np.linspace(-10,10, 50)
    # ottengo coppie (x,y)
    xx,yy = np.meshgrid(B0,B1, indexing = 'xy')
    z = np.zeros((B0.shape[0], B1.shape[0]))

    #ndenumerate itera sulle multidimensioni
    for (i,j), v in np.ndenumerate(z):  #i è la coppia di coordinate e j è il valore di z
        z[i,j],_ = costVectorial(x,y, theta = [[xx[i,j]], [yy[i,j]]])
    fig = plt.figure(figsize = (16,6))

    ax1 = fig.add_subplot(121)
    ax1.contour(xx,yy,z, np.logspace(-2,3,20))
    ax1.set_xlabel(r'$\theta_0$', fontsize ='18')
    ax1.set_ylabel(r'$\theta_1$', fontsize ='18')

    ax2 = fig.add_subplot(122, projection = '3d')
    ax2.plot_surface(xx,yy,z, cmap=cm.coolwarm)
    ax2.set_xlabel(r'$\theta_0$', fontsize ='18')
    ax2.set_ylabel(r'$\theta_1$', fontsize ='18')
    ax2.set_zlabel(r'$j(\theta_0,\theta_1)$', fontsize ='18')

    plt.show()
    
    # x con la colonna di 1 già aggiunta
def linear_regression_scikit(x,y):
    linear_reg = LinearRegression()
    linear_reg.fit(x,y)
    theta_sklearn = np.array([linear_reg.intercept_[0], linear_reg.coef_[0][1]]).reshape(-1,1)
    return theta_sklearn

def feature_scaling(design_matrix):
    
    Z = np.copy(design_matrix)
    
    mu = np.mean(Z, axis = 0) #axis = 0 lo fa per le colonne
    std = np.std(Z, axis = 0)
    
    Z = (Z - mu)/std
 
    return Z

def feature_scaling_max_value(X, columns):
    for column in columns:
        X[[column]] = X[[column]]/X[column].max()
    return X

def plot_correlation(dataframe, method = 'pearson'):
    f, (ax1) = plt.subplots(1,1, figsize = (15,8))
    correlation_matrix = dataframe.corr(method = method)
    sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm_r', ax=ax1, linewidths=0.2, vmin=-1, vmax=1)
    plt.show()

def polinomyal_features(dataFrame, degree, column):
    
    new_dataFrame = dataFrame.copy()
    for i in range(1,degree):
        column_name = '('+column + ')**{}'.format(i+1)
        new_dataFrame[column_name] = dataFrame[column]**(i+1)

    return new_dataFrame

def polynomial_regression(dataFrame, Y, selected_feature, degree  ):
    X = dataFrame[[selected_feature]]
    x_poly_scaled = feature_scaling(polinomyal_features(X, degree, selected_feature))
    x_poly_scaled_int = np.c_[np.ones((x_poly_scaled.shape[0],1)), x_poly_scaled]
    theta_poly, cost_history_vectorized, _, _ = gradientDescentVectorized(x_poly_scaled_int, Y, theta = np.zeros((x_poly_scaled_int.shape[1], 1)), alpha = 0.01 , max_iters = 150000, early=False, epsilon=0.00001)
    X_plot, Y_plot = zip(*sorted(zip(X.values,x_poly_scaled_int.dot(theta_poly))))
    plt.figure(figsize=(20,12))
    plt.scatter(X, Y, marker='x', c='r')
    plt.plot(X_plot, Y_plot, label='Polynomial hypothesis')
    plt.xlabel(selected_feature, fontsize= 15)
    plt.ylabel('Y', fontsize= 15)
    plt.legend(loc = 'best')
    plt.show()
    return theta_poly

def polynomial_regression_max_scaling(dataFrame, Y, selected_feature, degree  ):
    X = dataFrame[[selected_feature]]
    X2 = X.copy()
    X = polinomyal_features(X, degree, selected_feature)
    x_poly_scaled = feature_scaling_max_value(X, X.columns)
    x_poly_scaled_int = np.c_[np.ones((x_poly_scaled.shape[0],1)), x_poly_scaled]
    theta_poly, cost_history_vectorized, _, _ = gradientDescentVectorized(x_poly_scaled_int, Y, theta = np.zeros((x_poly_scaled_int.shape[1], 1)), alpha = 0.01 , max_iters = 150000, early=False, epsilon=0.00001)
    X_plot, Y_plot = zip(*sorted(zip(X2.values,x_poly_scaled_int.dot(theta_poly))))
    plt.figure(figsize=(20,12))
    plt.scatter(X2, Y, marker='x', c='r')
    plt.plot(X_plot, Y_plot, label='Polynomial hypothesis')
    plt.xlabel(selected_feature, fontsize= 15)
    plt.ylabel('Y', fontsize= 15)
    plt.legend(loc = 'best')
    plt.show()
    return theta_poly
    
def linear_regression(x, y):
    x = np.c_[np.ones(x.shape[0]), x]
    theta_gd, cost_history, _, theta_history = gradientDescentVectorized(x, y, theta = np.zeros((x.shape[1], 1)))
    return theta_gd, cost_history

def normal_equation(X,Y):
    elapsing_time = 0
    elapsing_time -= time.time()
    
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)
    
    elapsing_time += time.time()
     
    return theta, elapsing_time


def reg_lin_normal_equation(X,Y):
    X_int = np.c_[np.ones((X.shape[0],1)),X]
    theta_linear, _ = normal_equation(X_int,Y)
    return theta_linear

def polynomial_regression_normal_equation(X, Y, selected_feature, degree  ):
    X_poly = polinomyal_features(X, degree,selected_feature);
    X_poly_int = np.c_[np.ones((X_poly.shape[0],1)), X_poly]
    theta_poly, _ = normal_equation(X_poly_int,Y)
    X_plot, Y_plot = zip(*sorted(zip(X.values,X_poly_int.dot(theta_poly))))
    plt.figure(figsize=(20,12))
    plt.scatter(X, Y, marker='x', c='r')
    plt.plot(X_plot, Y_plot, label='Polynomial hypothesis')


    plt.xlabel(selected_feature, fontsize= 15)
    plt.ylabel('Y', fontsize= 15)
    plt.legend(loc = 'best')
    plt.show()
