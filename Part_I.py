
##########################################################
#  Python script template for Question 1 (IAML Level 11)
#  Note that
#  - You should not change the name of this file, 'iaml212cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

#<----

# Q1.1
# Import packages
import os
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import graphviz
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score 
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from pandas.api.types import CategoricalDtype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import StratifiedKFold
from iaml_cw2_helpers import *
%matplotlib inline
print_versions()


#At first,spliting the data, the following:

X,Y=load_Q1_dataset(filename='data/dataset_q1.csv')
print('X:', X.shape, 'Y:', Y.shape) 
Xtrn = X[100: ,:]; Ytrn = Y[100:] # training data set
Xtst = X[0:100 ,:]; Ytst = Y[0:100]# test data set
# Q1.1

def iaml212cw2_q1_1(Xtrn,Ytrn):
    fig, ax = plt.subplots(figsize=(7,7))

    for i in range(9):
        x=Xtrn[:,i]
        Xa=x[Ytrn==0]
        Xb=x[Ytrn==1]
        plt.subplot(3,3,i+1)
        plt.hist([Xa, Xb],bins=15)
        plt.title('A'+str(i),fontsize=12)
        plt.legend(['Xa', 'Xb'])
        plt.grid(True) # Enables grid
    plt.tight_layout()
    plt.show()
# Runing this function, it returns a total of nine figures in a 3-by-3 grid 
# where each feature is aboout how Xtrn is distributed for each class.


## Q1.2

def iaml212cw2_q1_2(Xtrn,Ytrn):
    corr=[]
    for i in range(9):
        x=Xtrn[:,i]
        der_x=x-np.mean(x)
        der_y=Ytrn-np.mean(x)
        r=np.dot(der_x,der_y)/np.sqrt(np.dot(der_x,der_x)* np.dot(der_y,der_y))
        corr.append(round(r,6)) 
    
    return corr


# Runing this function, it returns a list of correlation coefficients 
#between each attribute of Xtrn and the label Ytrn.


## Q1.4
def iaml212cw2_q1_4():
    sample_var=[]
    for i in range(9):
        x=Xtrn[:,i]
        der_x=x-np.mean(x)
        n=Xtrn.shape[0]
        sv=np.dot(der_x,der_x)/(n-1)
        sample_var.append(sv)
        sum_variance=sum_list(sample_var)
    print("Amount of variance explained by each attribute in decreasing:",sorted(sample_var,reverse = True))
    print("Sum of variance explained by all attributes:",sum_list(sorted(sample_var,reverse = True)))
        ##(b) 
        ##1)
    s=pd.Series(sample_var, index=['A0','A1','A2','A3','A4','A5','A6','A7','A8'])
    s=s.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8,4))

    plt.subplot(1,2,1)    

    plt.plot(s.index, s.values)
    plt.title('Amount of variance explained by each attribute')
    plt.xlabel('Attributes')
    plt.ylabel('Variance')
    plt.grid()
        
        ##2

    cv=list()
    cu=0
    for i in range(9):
        cu +=s.values[i]
        ratio=cu/sum_variance
        cv.append(ratio)

    plt.subplot(1,2,2)    
    plt.plot(s.index, np.array(cv))
    plt.title('Cumulative variance ratio vs Number of Attributes')
    plt.xlabel('(sorted) Attributes')
    plt.ylabel('Cumulative variance ratio')
    plt.grid()


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.7)

    plt.show()

iaml212cw2_q1_4()

##Runing this function, it returns the unbiased sample variance of each attribute of Xtrn in decreasing order,
#the sum of all the variances, and two graphs in (b).

## Q1.5
def iaml212cw2_q1_5():
    pca=PCA()
    pca.fit(Xtrn)
    #a
    sum_var=sum_list(pca.explained_variance_)
    print('The total amount of unbiased sample variance explained by the whole set of principal components.:', sum_var)
    #b 
    fig, ax = plt.subplots(figsize=(8,4))
    plt.subplot(1,2,1)   
    
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Amount of Variance Explained')
    plt.grid()
    
    plt.subplot(1,2,2)   
    
    PC_values = np.arange(pca.n_components_) + 1
    out_sum = np.cumsum(pca.explained_variance_ratio_)  
    plt.plot(PC_values,out_sum,'bo-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative variance ratio')
    plt.grid()
    
    plt.show()
    
    #c
    p1=[]
    p2=[]
    for i in range(Xtrn.shape[0]):
        p1.append(np.dot(pca.components_[0],Xtrn[i,:]))
        p2.append(np.dot(pca.components_[1],Xtrn[i,:]))
               
    X2d=np.array(list(zip(p1,p2)))
    sub_labels = [0,1]
    sub_cats = ['No','Yes']

# --- Basic Plot --- #
    plt.figure(figsize=(12,8))
    for label, cat in zip(sub_labels, sub_cats):
        plt.scatter(X2d[Ytrn== label, 0], X2d[Ytrn == label, 1], alpha=.5, lw=2, label=cat)
    plt.axis('equal')
    plt.legend(loc='center left', scatterpoints=3, bbox_to_anchor=[1.01, 0.5])
    plt.title('Labelled data in PCA space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    top_plot = plt.gca()
    plt.grid()
    plt.show()          
               
               
               
    #d          
    corr1=[]
    corr2=[]
    for i in range(9):
        x=Xtrn[:,i]
        der_x=x-np.mean(x)
        der_p1=p1-np.mean(p1)
        r1=np.dot(der_x,der_p1)/np.sqrt(np.dot(der_x,der_x)* np.dot(der_p1,der_p1))
        corr1.append(round(r,6)) 
        der_p2=p2-np.mean(p2)
        r2=np.dot(der_x,der_p2)/np.sqrt(np.dot(der_x,der_x)* np.dot(der_p2,der_p2))
        corr2.append(round(r,6)) 
    print(corr1)
    print(corr2)
    
    
    
# Q1.6
def iaml212cw2_q1_6():
    scaler = StandardScaler ().fit (Xtrn)
    Xtrn_s = scaler.transform(Xtrn) # standardised training data 
    Xtst_s = scaler.transform(Xtst) # standardised test data
    c_s=np.logspace(0.01, 100, num=13)
    skf = StratifiedKFold(n_splits=5)
    c_avgcore=[]
    c_std_core=[]
    c_avgcore_train=[]
    c_std_core_train=[]
    for i in range(13):
        svc_rbf = SVC(C=c_s[i],kernel='rbf')
        acc_score_each_fold = []
        acc_score_each_fold_train=[]
        for train_index , test_index in skf.split(Xtrn_s,Ytrn):
            X_train,X_test = Xtrn_s[train_index,:],Xtrn_s[test_index,:]
            y_train,y_test = Ytrn[train_index] , Ytrn[test_index]
            svc_rbf.fit(X_train,y_train)
            pred_values =svc_rbf.predict(X_test)
            acc = accuracy_score(pred_values,y_test)
            acc_score_each_fold.append(acc)
            c_avgcore.append(sum_list(acc_score_each_fold)/5)
            c_std_core.append(np.std(acc_score_each_fold))
            pred_values_train =svc_rbf.predict(X_train)
            acc_train = accuracy_score(pred_values_train,y_train)
            acc_score_each_fold_train.append(acc_train)
            c_avgcore_train.append(sum_list(acc_score_each_fold_train)/5)
            c_std_core_train.append(np.std(acc_score_each_fold_train))
    
    fig, ax = plt.subplots()
    ax.plot(c_s, c_avgcore, 'g-', label=r'$mean$')
    ax.plot(c_s, c_std_core, 'y-', label=r'$sd$')
    ax.plot(c_s, c_avgcore_train, 'g-', label=r'$train_mean$')
    ax.plot(c_s, c_std_core_train, 'y-', label=r'$train_sd$')
    ax.legend(loc='upper right', fontsize=10)
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    svc_rbf = SVC(C=c_s[0],kernel='rbf')
    svc_rbf.fit(Xtst_s,Ytst)
    pred_values =svc_rbf.predict(Xtst_s)
    acc = accuracy_score(pred_values,Ytst)
    sum=0
    for i in range(pred_values.shape[0]):
        if pred_values[i]==Ytst[i]:
            sum +=1
    print(sum,acc)
     
# Q1.8
def iaml212cw2_q1_8():
    A4=Xtrn[:,4]
    A7=Xtrn[:,7]
    a7=A7[A4>=1]
    a4=A4[A4>=1]
    y_trn=Ytrn[A4>=1]
    y_trn_0=y_trn[y_trn==0]
    a_4=a4[y_trn==0]
    a_7=a7[y_trn==0]
    Ztrn=np.stack((a_4, a_7), axis=0)
    return np.cov(Ztrn),np.std(a_4, ddof=1),np.std(a_7, ddof=1) 

# Q1.9
def iaml212cw2_q1_9():
#
# iaml212cw2_q1_9()   # comment this out when you run the function

# Q1.10
def iaml212cw2_q1_10():
#
# iaml212cw2_q1_10()   # comment this out when you run the function

# Q1.11
def iaml212cw2_q1_11():
    skf = StratifiedKFold(n_splits=5)
    model = LogisticRegression(max_iter=1000,random_state=0)
    acc_score = []
    for train_index , test_index in skf.split(Xtrn, Ytrn):
        X_train , X_test = Xtrn[train_index,:],Xtrn[test_index,:]
        y_train , y_test = Ytrn[train_index] , Ytrn[test_index]
     
        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
     
        acc = accuracy_score(pred_values , y_test)
        acc_score.append(acc)
     
    avg_acc_score = sum(acc_score)/5
    avg_acc_sd=np.std(acc_score)
 

    print('Mean of accuracy: {}'.format(avg_acc_score))
    print('Standard deviation of accuracy:{}'.format(avg_acc_sd))  
    
    avg_acc_score_list=[]
    avg_acc_sd_list=[]

    for i in range(9):
        x_trian=np.copy(Xtrn)
        xtr_drop=np.delete(x_trian,i,1)
        acc_score =[]
    
        for train_index , test_index in skf.split(xtr_drop, Ytrn):
            X_train , X_test = xtr_drop[train_index,:],xtr_drop[test_index,:]
            y_train , y_test = Ytrn[train_index] , Ytrn[test_index]
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            acc = accuracy_score(pred_values , y_test)
            acc_score.append(acc)
    avg_acc_score_list.append(sum(acc_score)/5)
    avg_acc_sd_list.append(np.std(acc_score))
    
#
# iaml212cw2_q1_11()   # Return the answer for question 11