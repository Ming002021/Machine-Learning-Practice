
##########################################################
#  Python script template for Question 2 (IAML Level 11)
#  Note that
#  - You should not change the filename of this file, 'iaml212cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

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

#<----

Xtrn_org , Ytrn_org , Xtst_org , Ytst_org = load_Q2_dataset ( ) 
Xtrn = Xtrn_org / 255.0
Xtst = Xtst_org / 255.0
Ytrn = Ytrn_org -1
Ytst = Ytst_org -1
Xmean = np.mean(Xtrn, axis=0)
Xtrn_m =Xtrn -Xmean
Xtst_m = Xtst -Xmean # Mean−normalised versions



# Q2.1
def iaml212cw2_q2_1():
    print(Xtrn.min(),Xtrn.max(),np.mean(Xtrn),np.std(Xtrn),Xtst.min(),Xtst.max(),np.mean(Xtst),np.std(Xtst))
    plt.subplot(1,2,1)
    X0 = 1-Xtrn[0].reshape(28,28) # sample 2D array 
    plt.imshow(X0, cmap="gray") 
    plt.show() 

    plt.subplot(1,2,2)
    X1 = 1-Xtrn[1].reshape(28,28) # sample 2D array 
    plt.imshow(X1, cmap="gray") 
 
    plt.show() 
#
# iaml212cw2_q2_1()   # return the minimum, maximum, mean, and standard deviation of pixel values for each Xtrn and Xtst and the images of the first instances.

# Q2.3
def iaml212cw2_q2_3():
    Xtrn_0=Xtrn[Ytrn==0]
    Xtrn_5=Xtrn[Ytrn==5]
    Xtrn_8=Xtrn[Ytrn==8]
 
    kmeans_3 = KMeans(n_clusters=3, random_state=1000)
    kmeans_5 = KMeans(n_clusters=5, random_state=1000)
    c30=kmeans_3.fit(Xtrn_0).cluster_centers_
    c35=kmeans_3.fit(Xtrn_5).cluster_centers_
    c38=kmeans_3.fit(Xtrn_8).cluster_centers_
    c50=kmeans_5.fit(Xtrn_0).cluster_centers_
    c55=kmeans_5.fit(Xtrn_5).cluster_centers_
    c58=kmeans_5.fit(Xtrn_8).cluster_centers_

##k=3
    k3=np.vstack((c30,c35,c38))

##k=5
    k5=np.vstack((c50,c55,c58))
    _, axs = plt.subplots(3,5, figsize=(8,8))
    axs = axs.flatten()
    for  x, ax in zip(k5, axs):
        img= 1-x.reshape(28,28)
        ax.imshow(img,cmap="gray")
    plt.show()
    _, axs = plt.subplots(3,3, figsize=(8, 8))
    axs = axs.flatten()
    for  x, ax in zip(k3, axs):
        img= 1-x.reshape(28,28)
        ax.imshow(img,cmap="gray")
    plt.show()

# iaml212cw2_q2_3()   # Display the images of cluster centres for each k

# Q2.5
def iaml212cw2_q2_5():
    lr = LogisticRegression(max_iter=1000,random_state=0)
    lr.fit(Xtrn, Ytrn)
    print('Classification accuracy on training set: {:.3f}'.format(lr.score(Xtrn, Ytrn)))
    print('Classification accuracy on test set: {:.3f}'.format(lr.score(Xtst,Ytst)))
    letter_miss_number={}
    pred_values = lr.predict(Xtst)
    for i in range(pred_values.shape[0]):
        if pred_values[i] != Ytst[i]:
            letter=Ytst[i]
            if letter not in letter_miss_number:
                letter_miss_number[letter]=1
            else:
                letter_miss_number[letter] +=1
    print(letter_miss_number)

# Q2.6 
def iaml212cw2_q2_6():
#
# iaml212cw2_q2_6()   # comment this out when you run the function

# Q2.7 
def iaml212cw2_q2_7():

    Xtrn_m_0=Xtrn_m[Ytrn==0]
    cov_mat=np.cov(Xtrn_m_0)
    mean_vec=Xtrn_m_0.mean(0)

    cov_mat_dia=cov_mat.diagonal()

    print(round(cov_mat_dia.min(),6),round(cov_mat_dia.max(),6),round(np.mean(cov_mat_dia),6))
    print(round(np.std(cov_mat_dia),6))
    print(round(cov_mat.min(),6),round(cov_mat.max(),6),round(np.mean(cov_mat),6))
    print(round(np.std(cov_mat),6))
    plt.hist(cov_mat_dia,bins=15, density=True, facecolor='g', alpha=0.75)
    plt.title('Diagonal values of the covariance matrix.')
    plt.grid()  
    
    plt.show()
    from scipy.stats import multivariate_normal
    Xtrn_m0=Xtrn_m[Ytrn==0]
    Xtrn_m_0=Xtrn_m0.T
    Xtst_m_0=Xtst_m.T

    cov_mat=np.cov(np.transpose(Xtrn_m0))
    mean_vec=Xtrn_m0.mean(0)

    multivariate_normal.pdf(Xtrn_m0[0], mean_vec, cov_mat)


  


# Q2.8
def iaml212cw2_q2_8():

    gm = GaussianMixture(n_components=1, covariance_type='full')
    gm.fit(Xtrn_m_0)
    gm.score_samples(Xtst_m)[0]
    acc_train=[]
    acc_test=[]

    for i in range(0,26):
        Xtrn_m_class=Xtrn_m[Ytrn==i]
        Ytrn_class=Ytrn[Ytrn==i]
        gm.fit(Xtrn_m_class)
        pred_value_train=gm.predict(Xtrn_m_class)
        Xtst_m_class=Xtst_m[Ytst==i]
        Ytst_class=Ytst[Ytst==i]
        pred_value_test=gm.predict(Xtst_m_class)
        acc_train.append(accuracy_score(pred_value_train,Ytrn_class))
        acc_test.append(accuracy_score(pred_value_test,Ytst_class))
    



#
# iaml212cw2_q2_8()   # comment this out when you run the function

# Q2.10 
def iaml212cw2_q2_10():
    a=[]
    b=[]


    for c in [1,2,4,8]:
        gm= GaussianMixture(n_components=c, random_state=0 )
        gm.fit(Xtrn_m)
        pred_train=gm.predict(Xtrn_m)
        pred_test=gm.predict(Xtst_m)
        a.append(accuracy_score(pred_train,Ytrn))
        b.append(accuracy_score(pred_test,Ytst))
    
    print(a,b)
    
    gm= GaussianMixture(n_components=2, random_state=0 )
    gm.fit(Xtst_m)
 
    pred_test=gm.predict(Xtst_m)
 
    ac=accuracy_score(pred_test,Ytst)
    print(round(ac,6))
    gm.get_params()
#
# iaml212cw2_q2_10()   # comment this out when you run the function
