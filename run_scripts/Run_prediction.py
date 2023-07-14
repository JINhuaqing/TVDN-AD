#!/usr/bin/env python
# coding: utf-8

# This file contains the code for ND, DMD, TVDMD and TVDN including
# 
# - regression:  double reg, first for selecting feaures and second for AUC and prediction
# 
# 
# Here I split the dataset into testing and training sets for prediction

# In[34]:


import sys
sys.path.append("/home/huaqingj/MyResearch/TVDN-AD/")


# In[35]:


import importlib
from pyTVDN import TVDNDetect

import pyTVDN.utils
importlib.reload(pyTVDN.utils)
from pyTVDN.utils import load_pkl, save_pkl

from sklearn.cluster import KMeans
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import os
from scipy import signal
import pickle
import seaborn as sns
#from tqdm.autonotebook import tqdm
from tqdm import tqdm
import numbers


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')


# ### save fns

# In[37]:


os.chdir("/home/huaqingj/MyResearch/TVDN-AD")
resDir = Path("./results")
dataDir = Path("./data")


# In[38]:


def split_data(X_data, Y_data, ratio=0.5):
    sel_idx = np.sort(np.random.choice(len(Y_data), int(len(Y_data)*ratio), replace=False))
    X_train, Y_train = X_data[sel_idx, :], Y_data[sel_idx]
    X_test, Y_test = np.delete(X_data, sel_idx, axis=0), np.delete(Y_data, sel_idx)
    return X_train, Y_train, X_test, Y_test


# In[39]:


def TuningCFn(inpX, inpY, Cs=[0.1, 0.2, 0.4, 0.8, 1, 1.6, 3.2, 6.4, 12.8, 25.6], penalty="l2"):
    aucCs = []
    for C in Cs:
        eProbs = []
        loo = LeaveOneOut()
        for trIdxs, testIdxs in loo.split(inpX):
            clf = LogisticRegression(penalty=penalty, random_state=0, C=C)
            clf.fit(inpX[trIdxs, :], inpY[trIdxs])
            eProbs.append(clf.predict_proba(inpX[testIdxs, :]))
        eProbs = np.array(eProbs).squeeze()
        auc = roc_auc_score(inpY, eProbs[:, 1])
        fpr, tpr, thresholds = roc_curve(inpY, eProbs[:, 1], pos_label=1)
        aucCs.append(auc)
            
    optC = Cs[np.argmax(aucCs)]
    res = edict()
    res["optC"] = optC
    res["Cs"] = Cs
    res["aucCs"] = aucCs
    return res


# ### Parameters

# In[40]:


Cs = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.6, 3.2, 6.4, 12.8, 25.6]
freq = 120
penalty = "l2"


# In[41]:


model = "TVDN"
print(f"Run for {model}")

cur_data = load_pkl(resDir/f"./tmp_res/{model}_data.pkl")
stdXs, Ys = cur_data["stdXs"], cur_data["Ys"]


# ### RUN

# In[ ]:


# you can use DMD, ND or TVDMD
np.random.seed(0)

aucs = []
mAUCs = []
stdAUCs = []
for ix in tqdm(range(100)):

    X_train, Y_train, X_test, Y_test = split_data(stdXs, Ys)
    
    # training to selected features
    ## fit
    gOptC1 = TuningCFn(X_train, Y_train, Cs, penalty)["optC"]
    
    # fit the first reg
    clfFinal = LogisticRegression(penalty=penalty, random_state=0, C=gOptC1)
    clfFinal.fit(X_train, Y_train)
    coefsFinal = clfFinal.coef_.reshape(-1)
    
    # boostrap for pval
    repTime = 10000
    parassBoot = []
    for i in range(repTime):
        bootIdx = np.random.choice(len(Y_train), len(Y_train))
        Y_trainBoot = Y_train[bootIdx]
        X_trainBoot = X_train[bootIdx]
        clf = LogisticRegression(penalty=penalty, random_state=0, C=gOptC1)
        clf.fit(X_trainBoot, Y_trainBoot)
        parasBoot = clf.coef_.reshape(-1)
        parassBoot.append(parasBoot)
        
    parassBoot = np.array(parassBoot)
    lows, ups = coefsFinal-parassBoot.std(axis=0)*1.96, coefsFinal+parassBoot.std(axis=0)*1.96
    kpidxBoot = np.bitwise_or(lows >0,  ups < 0)
    select_fs_idx = kpidxBoot


    # testing to get prediction
    X_test_selected = X_test[:, select_fs_idx]
    
    gOptC2 = TuningCFn(X_test_selected, Y_test, Cs, penalty=penalty)["optC"]
    
    eProbs = []
    loo = LeaveOneOut()
    parass = []
    for trIdxs, testIdxs in loo.split(X_test_selected):
        curOptC = TuningCFn(X_test_selected[trIdxs, :], Y_test[trIdxs], Cs, penalty)["optC"]
        clf = LogisticRegression(penalty=penalty, random_state=0, C=curOptC)
        clf.fit(X_test_selected[trIdxs, :], Y_test[trIdxs])
        paras = np.concatenate([clf.intercept_, clf.coef_.reshape(-1)])
        parass.append(paras)
        eProbs.append(clf.predict_proba(X_test_selected[testIdxs, :]))
    eProbs = np.array(eProbs).squeeze()
    auc = roc_auc_score(Y_test, eProbs[:, 1])
    fpr, tpr, thresholds = roc_curve(Y_test, eProbs[:, 1], pos_label=1)
    parass = np.array(parass)
    
    
    nobs = X_test_selected.shape[0]
    Aucss = []
    for j in range(10000):
        flag = 1 # to avoid all 1 or all 0 cases
        while flag:
            testIdx = np.random.choice(nobs, int(nobs/5), False)
            trainIdx = np.delete(np.arange(nobs), testIdx)
            n_test, n_train = np.sum(Y_test[testIdx]), np.sum(Y_test[trainIdx])
            flag = (n_test==0) + (n_train==0) + (n_test==len(testIdx)) + (n_train==len(trainIdx))
            
        clf = LogisticRegression(penalty=penalty, random_state=0, C=gOptC2)
        clf.fit(X_test_selected[trainIdx], Y_test[trainIdx])
        curEprobs = clf.predict_proba(X_test_selected[testIdx, :])
        curAuc = roc_auc_score(Y_test[testIdx], curEprobs[:, 1])
        Aucss.append(curAuc)
    mAUC = np.mean(Aucss)
    stdAUC = np.std(Aucss)
    
    aucs.append(auc)
    mAUCs.append(mAUC)
    stdAUCs.append(stdAUC)

resROC = {"aucs":aucs, 
         "mAUCs":mAUCs, 
         "stdAUCs":stdAUCs}
save_pkl(resDir/f"{model}_prediction_{ix+1}.pkl", resROC)


# In[ ]:




