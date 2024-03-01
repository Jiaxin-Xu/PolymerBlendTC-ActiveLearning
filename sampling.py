import numpy as np
import pandas as pd
import random

def uncertainty_sampling(clf, unknown_indexes, X_pool):
    pred_prob = clf.predict_proba(X_pool[unknown_indexes])
    ind = np.argmin(np.abs( 
        list(pred_prob[:,0] - pred_prob[:,1])
        ))
    return unknown_indexes[ind]

def certainty_sampling(clf, unknown_indexes, X_pool):
    pred_prob = clf.predict_proba(X_pool[unknown_indexes])
    ind = np.argmax(list(pred_prob[:,1]))
    return unknown_indexes[ind]

def random_sampling(clf, unknown_indexes, X_pool):
    ind = random.sample(range(len(unknown_indexes)), 1)[0]
    return unknown_indexes[ind]

def balanced_sampling(clf, unknown_indexes, X_pool):
    pred_prob = clf.predict_proba(X_pool[unknown_indexes])
    u = np.abs(list(pred_prob[:,0] - pred_prob[:,1]))
    c = pred_prob[:,1]
    alpha = 0.5
    ind = np.argmax(alpha*c - (1-alpha)*u)
    return unknown_indexes[ind]

# modification for multiple sampling at a time
def uncertainty_sampling_n(clf, unknown_indexes, X_pool,n=4):
    pred_prob = clf.predict_proba(X_pool[unknown_indexes])
    inds = np.argsort(np.abs( 
        list(pred_prob[:,0] - pred_prob[:,1])
        ))[:n]
    return unknown_indexes[inds]

def certainty_sampling_n(clf, unknown_indexes, X_pool, n=4):
    pred_prob = clf.predict_proba(X_pool[unknown_indexes])
    inds = np.argsort(list(pred_prob[:,1]))[-n:]
    return unknown_indexes[inds]

def random_sampling_n(clf, unknown_indexes, X_pool, n=4):
    inds = np.array(random.sample(range(len(unknown_indexes)), n))
    return unknown_indexes[inds]

###TODO
def balanced_sampling_n(clf, unknown_indexes, X_pool, n=4):
    pred_prob = clf.predict_proba(X_pool[unknown_indexes])
    # u = np.abs(list(pred_prob[:,0] - pred_prob[:,1]))
    # c = pred_prob[:,1]
    # alpha = 0.5
    # ind = np.argsort(alpha*c - (1-alpha)*u)[-n:]
    # use half from certainty and half from uncertainty
    n_c = int(n/2)
    inds_c = np.argsort(list(pred_prob[:,1]))[-n_c:]
    inds_u = np.argsort(np.abs( 
        list(pred_prob[:,0] - pred_prob[:,1])
        ))[:(n-n_c)]
    inds = np.concatenate((inds_c, inds_u))

    return unknown_indexes[inds]