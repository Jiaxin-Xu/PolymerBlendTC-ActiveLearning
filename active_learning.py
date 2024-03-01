from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np 
import pandas as pd
import random
import matplotlib.pyplot as plt
import sampling
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,make_scorer


def classifier_gridsearch(X_train, y_train, base_model=RandomForestClassifier(), cv=2, n_iter=10, random_state=42):
    # NOTE: cv=5 or more will invoke problems when #good is small at the very beginning
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 20)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num = 11)]
    # Create the random grid
    random_grid_rf = {'n_estimators': n_estimators,
                'max_depth': max_depth}
    precision_scorer = make_scorer(precision_score,zero_division=0)
    search = RandomizedSearchCV(estimator = base_model, refit=True,scoring =precision_scorer, param_distributions = random_grid_rf, n_iter = n_iter, cv = cv, verbose=0, random_state=random_state, n_jobs = -1)
    search.fit(X_train, y_train)

    return search

def findgood(X_pool, y_pool, sampling, train_indexes=None, unknown_indexes=None, ngoodfind=10):
    ngood = 0
    niter = 0
    # ngood_ls = [0]
    niter_ls = [0]
    while ngood <ngoodfind: # target find 10 good PB
        X_train = X_pool[train_indexes]
        y_train = y_pool[train_indexes]
        clf = classifier_gridsearch(X_train, y_train)

        n = sampling(clf, unknown_indexes, X_pool)
        unknown_indexes.remove(n)
        train_indexes.append(n)
        niter += 1
        if y_pool[n] == 1:
            ngood += 1
            # ngood_ls.append(ngood)
            niter_ls.append(niter)
    print(f'Iter {niter}, {ngood} good PB found!')
    return niter_ls

def findgood_n(X_pool, y_pool, sampling, nsamp=4,train_indexes=None, unknown_indexes=None, ngoodfind=10):
    ngood = 0
    niter = 0
    # ngood_ls = [0]
    niter_ls = [0]
    # ngood_ls = [0]
    iter_score = []
    while ngood <ngoodfind: # target find 10 good PB
        X_train = X_pool[train_indexes]
        y_train = y_pool[train_indexes]
        clf = classifier_gridsearch(X_train, y_train)

        iter_score.append(clf.best_score_)
        print('The best CV score is: ',clf.best_score_)

        samp = sampling(clf, unknown_indexes, X_pool, n=nsamp) #returns an np.array
        # unknown_indexes.remove(n)
        # train_indexes.append(n)
        unknown_indexes = np.setdiff1d(unknown_indexes,samp)
        train_indexes = np.append(train_indexes,samp)

        niter += 1
        # if sum(y_pool[samp]) >= 1:
        ngood += sum(y_pool[samp])
        # ngood_ls.append(ngood)
        for i in range(sum(y_pool[samp])):
            if len(niter_ls) > ngoodfind:
                break
            niter_ls.append(niter)
        # ngood_ls.append(ngood)
            
    print(f'Iter {niter}, {ngood} good PB found!')
    return niter_ls, iter_score

def al_iter(X_pool, y_pool, niter=3, ninit=10, ngoodfind=10):
    # ceate an empty dataframe
    result_us,result_cs,result_bs,result_rs = np.zeros(shape=(ngoodfind+1,niter)),np.zeros(shape=(ngoodfind+1,niter)),np.zeros(shape=(ngoodfind+1,niter)),np.zeros(shape=(ngoodfind+1,niter))
    fail = 0
    i = 0
    while i < niter:
        # start with initial=10
        train_indexes = random.sample(range(len(y_pool)), ninit) #return a list
        unknown_indexes = [e for e in list(range(len(y_pool))) if e not in train_indexes] #return a list

        train_indexes_us,train_indexes_cs,train_indexes_bs,train_indexes_rs = train_indexes.copy(),train_indexes.copy(),train_indexes.copy(),train_indexes.copy()
        unknown_indexes_us,unknown_indexes_cs,unknown_indexes_bs,unknown_indexes_rs=unknown_indexes.copy(),unknown_indexes.copy(),unknown_indexes.copy(),unknown_indexes.copy()

        if sum(y_pool[train_indexes]) <= 1: # >= cv fold=2
            fail += 1
            print(f"Initialization{i} Failed! Redo!")
            continue
        
        print(f"Initialization[{i}]: {sum(y_pool[train_indexes])}/{len(train_indexes)} good polymers!")

        result_us[:,i] = findgood(X_pool=X_pool, y_pool=y_pool, sampling=sampling.uncertainty_sampling, train_indexes=train_indexes_us, unknown_indexes=unknown_indexes_us,ngoodfind=ngoodfind)
        result_cs[:,i] = findgood(X_pool=X_pool, y_pool=y_pool, sampling=sampling.certainty_sampling, train_indexes=train_indexes_cs, unknown_indexes=unknown_indexes_cs, ngoodfind=ngoodfind)
        result_bs[:,i] = findgood(X_pool=X_pool, y_pool=y_pool, sampling=sampling.balanced_sampling, train_indexes=train_indexes_bs, unknown_indexes=unknown_indexes_bs, ngoodfind=ngoodfind)
        result_rs[:,i] = findgood(X_pool=X_pool, y_pool=y_pool, sampling=sampling.random_sampling, train_indexes=train_indexes_rs, unknown_indexes=unknown_indexes_rs, ngoodfind=ngoodfind)

        i += 1
    
    print(f"Finish {niter}/{niter+fail} successful random initialization")
    return {'Uncertainty':result_us,'Certainty':result_cs,'Balanced':result_bs,'Random':result_rs}

def al_iter_n(X_pool, y_pool, nsamp=4,niter=3, ninit=10, ngoodfind=10):
    # ceate an empty dataframe
    result_us,result_cs,result_bs,result_rs = np.zeros(shape=(ngoodfind+1,niter)),np.zeros(shape=(ngoodfind+1,niter)),np.zeros(shape=(ngoodfind+1,niter)),np.zeros(shape=(ngoodfind+1,niter))
    iter_score_us,iter_score_cs,iter_score_bs,iter_score_rs = [],[],[],[]
    fail = 0
    i = 0
    while i < niter:
        # start with initial=10
        train_indexes = np.array(random.sample(range(len(y_pool)), ninit)) #return a list
        unknown_indexes = np.array([e for e in list(range(len(y_pool))) if e not in train_indexes]) #return a list

        train_indexes_us,train_indexes_cs,train_indexes_bs,train_indexes_rs = train_indexes.copy(),train_indexes.copy(),train_indexes.copy(),train_indexes.copy()
        unknown_indexes_us,unknown_indexes_cs,unknown_indexes_bs,unknown_indexes_rs=unknown_indexes.copy(),unknown_indexes.copy(),unknown_indexes.copy(),unknown_indexes.copy()

        if sum(y_pool[train_indexes]) <= 1: # >= cv fold=2
            fail += 1
            print(f"Initialization{i} Failed! Redo!")
            continue
        
        print(f"Initialization[{i}]: {sum(y_pool[train_indexes])}/{len(train_indexes)} good polymers!")

        result_us[:,i],score_us = findgood_n(X_pool=X_pool, y_pool=y_pool, sampling=sampling.uncertainty_sampling_n, nsamp=nsamp, train_indexes=train_indexes_us, unknown_indexes=unknown_indexes_us,ngoodfind=ngoodfind)
        result_cs[:,i], score_cs = findgood_n(X_pool=X_pool, y_pool=y_pool, sampling=sampling.certainty_sampling_n, nsamp=nsamp, train_indexes=train_indexes_cs, unknown_indexes=unknown_indexes_cs, ngoodfind=ngoodfind)
        result_bs[:,i], score_bs = findgood_n(X_pool=X_pool, y_pool=y_pool, sampling=sampling.balanced_sampling_n, nsamp=nsamp, train_indexes=train_indexes_bs, unknown_indexes=unknown_indexes_bs, ngoodfind=ngoodfind)
        result_rs[:,i], score_rs = findgood_n(X_pool=X_pool, y_pool=y_pool, sampling=sampling.random_sampling_n, nsamp=nsamp, train_indexes=train_indexes_rs, unknown_indexes=unknown_indexes_rs, ngoodfind=ngoodfind)

        iter_score_us.append(score_us),iter_score_cs.append(score_cs),iter_score_rs.append(score_rs),iter_score_bs.append(score_bs)
        i += 1
    
    print(f"Finish {niter}/{niter+fail} successful random initialization")
    return {'Uncertainty':[result_us,iter_score_us],'Certainty':[result_cs,iter_score_cs],'Random':[result_rs,iter_score_rs],'Balanced':[result_bs,iter_score_bs]}


def active_learning_plot(ngoodfind,al_iter_results,nsamp=4,version='v0'):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.style.use('seaborn-whitegrid')
    ngood_ls = list(range(ngoodfind+1))
    ax.errorbar(al_iter_results['Uncertainty'][0].mean(axis=1), ngood_ls, xerr=al_iter_results['Uncertainty'][0].std(axis=1),fmt = '-bo',capsize=6,label='Uncertainty Sampling')
    ax.errorbar(al_iter_results['Certainty'][0].mean(axis=1), ngood_ls, xerr=al_iter_results['Certainty'][0].std(axis=1),fmt = '-gv',capsize=6,label='Certainty Sampling')    
    ax.errorbar(al_iter_results['Balanced'][0].mean(axis=1), ngood_ls, xerr=al_iter_results['Balanced'][0].std(axis=1),fmt = '-cx',capsize=6,label='Balanced Sampling')
    ax.errorbar(al_iter_results['Random'][0].mean(axis=1), ngood_ls, xerr=al_iter_results['Random'][0].std(axis=1),fmt = '-rs',capsize=6,label='Random Sampling')
    ax.set_xlabel('Number of virtual experiments')
    ax.set_ylabel('Number of good polymers')
    ax.legend()
    fig.savefig(f"./figure/al_{version}_{ngoodfind}good_{nsamp}_precision_wBalance_v5.png")

def active_learning_score_plot(ngoodfind,niter,al_iter_results,nsamp=4,version='v0'):   
    df_us = get_score_df(al_iter_results['Uncertainty'][1],niter=niter)
    df_cs = get_score_df(al_iter_results['Certainty'][1],niter=niter)
    df_rs = get_score_df(al_iter_results['Random'][1],niter=niter)
    df_bs = get_score_df(al_iter_results['Balanced'][1],niter=niter)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.style.use('seaborn-whitegrid')
    ax.errorbar(list(range(len(df_us))),df_us.mean(axis=1),yerr=df_us.std(axis=1), fmt='-bo',capsize=6,label='Uncertainty Sampling')
    ax.errorbar(list(range(len(df_cs))),df_cs.mean(axis=1),yerr=df_cs.std(axis=1), fmt='-gv',capsize=6,label='Certainty Sampling')
    ax.errorbar(list(range(len(df_bs))),df_bs.mean(axis=1),yerr=df_bs.std(axis=1), fmt='-cx',capsize=6,label='Balanced Sampling')
    ax.errorbar(list(range(len(df_rs))),df_rs.mean(axis=1),yerr=df_rs.std(axis=1), fmt='-rs',capsize=6,label='Random Sampling')
 
    ax.set_xlabel('Number of virtual experiments')
    ax.set_ylabel('Precision score')
    ax.legend()
    fig.savefig(f"./figure/al_iterscore_{version}_{ngoodfind}good_{nsamp}_precision_wBalance_v5.png")

def get_score_df(results, niter):
    df = pd.DataFrame(results[0])
    for i in range(1,niter):
        dff  = pd.DataFrame({i:results[i]})
        df = pd.concat([df,dff],axis=1)
    return df
        
