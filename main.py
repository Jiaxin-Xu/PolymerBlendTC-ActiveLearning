"""Run active learner on classification tasks
"""

import active_learning
from data_prepare import read_blend_data_PE

if __name__ == '__main__':
    # read the chosen pb dataset
    X_ws_PE, y, label = read_blend_data_PE(filename='../polymer_blend_combined_data_v3.csv',npb=384)
    # set goal
    ngoodfind = 20
    niter=10 # number of random initials -- emsemble average!
    ninit=10 # number of initial random-sample training data (starting point)
    # do active learning iterations
    nsamp=4
    dataversion='v3'
    results = active_learning.al_iter_n(niter=niter, ninit=ninit, nsamp=nsamp,ngoodfind=ngoodfind, X_pool=X_ws_PE, y_pool=label)
    # plot 
    active_learning.active_learning_plot(ngoodfind,results,nsamp=nsamp,version=dataversion)
    active_learning.active_learning_score_plot(ngoodfind,niter,results,nsamp=nsamp,version=dataversion)




