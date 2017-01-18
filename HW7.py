import numpy as np
import pandas as pd
import scipy as sp
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import tree
from sklearn import ensemble
from sklearn.cross_validation import KFold
import StringIO
import matplotlib
import pydot

#!pip install pydot

def plot_tree_boundary(x, y, model, title, ax):
    ax.scatter(x[y==1,0], x[y==1,1], c='green')
    ax.scatter(x[y==0,0], x[y==0,1], c='white')
    
    interval=np.arange(0,1,0.01)
    n=np.size(interval)
    x1, x2=np.meshgrid(interval, interval)
    x1=x1.reshape(-1,1)
    x2=x2.reshape(-1,1)
    xx=np.concatenate((x1, x2), axis=1)
    
    yy=model.predict(xx)
    yy=yy.reshape((n,n))
    
    x1=x1.reshape(n,n)
    x2=x2.reshape(n,n)
    ax.contourf(x1,x2, yy, alpha=0.1, cmap='Greens')
    
    ax.set_title(title)
    ax.set_xlabel('Latitute')
    ax.set_ylabel('Longitude')
    
    return ax
    
    