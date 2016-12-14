# -*- coding: utf-8 -*-
"""
Create Figuer, compare three algorithms across 

collecting resutls from nmi, ccr, ars, and modularity
"""
import SBMmodels as SBM
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import time
import itertools
import scipy
import numpy as np
import math
   
# MACRO parameter setting
rw_filename = 'sentences.txt'
emb_filename = 'emb.txt'
num_paths =10
length_path = 60
emb_dim =50
winsize = 8

def cal_metrics_3(labels, y_est_full):
    N = len(labels)
    acc_nmi = metrics.normalized_mutual_info_score(labels, y_est_full)
    acc_ars = metrics.adjusted_rand_score(labels, y_est_full)
    Conf = metrics.confusion_matrix(labels, y_est_full)
    r,c = scipy.optimize.linear_sum_assignment(-1*Conf)
    acc_ccr = float(Conf[r,c].sum())/float(N)
    return acc_nmi, acc_ccr, acc_ars
#def smooth_abp(G, r, lambda1, m, mp, niter, truelabel):
#    res_nmi = []
#    res_ccr = []
#    for i in range(niter):
#        labels_est = SBM.SBM_ABP(G, r, lambda1, m, mp)
#        nmi_abp, ccr_abp = cal_metrics_3(truelabel, labels_est)
#        res_nmi.append(nmi_abp)
#        res_ccr.append(ccr_abp)
#    return np.mean(res_nmi), np.mean(res_ccr),np.std(res_nmi), np.std(res_ccr)

def get_label_list(G):
    # only work for simulated graphs
    nodeslist = G.nodes()
    ln = [G.node[i]['community'] for i in nodeslist]
    nodeslist = [str(x) for x in nodeslist]
    return ln, nodeslist 
    
def update_a_res(arry, acc, alg, param, value, i):
    if alg not in arry:
        arry[alg] = {}
    key = param + '- ' + str(value)
    if key not in arry[alg]:
        arry[alg][key]={}
    arry[alg][key][i] = acc
    return arry
    
def summary_res(nmi_arry, ccr_arry, ars_arry, truelabel, label, alg, param, value, i):
    # alg: 'deep'/'sc'/'abp', 
    # param: 'c'/'N', 
    # value: the value of param,
    # i: the random iter    
    nmi, ccr, ars = cal_metrics_3(truelabel, label)
    print 'the NMI is:', nmi
    print 'the CCR is:', ccr
    nmi_arry = update_a_res(nmi_arry, nmi, alg, param, value, i)
    ccr_arry = update_a_res(ccr_arry, ccr, alg, param, value, i)
    ars_arry = update_a_res(ars_arry, ars, alg, param, value, i)
    return nmi_arry, ccr_arry, ars_arry

def plot_res_3(res):
    import matplotlib.pyplot as plt
    nmi = res[0]
    ccr = res[1]
    ars = res[2]
    tm = nmi['deep'].keys()
    param = tm[0].split('-')[0]
    x_array = [float(z.split('-')[1].strip()) for z in tm ]
    x_array = sorted(x_array)
    tm = [param + '- ' + str(v) for v in x_array]
    # get nmi for three algs, mean and std
    nmi_deep_mean = [np.mean(nmi['deep'][z].values()) for z in tm]
    nmi_sc_mean = [np.mean(nmi['sc'][z].values()) for z in tm] 
    nmi_abp_mean = [np.mean(nmi['abp'][z].values()) for z in tm] 
    nmi_deep_std = [np.std(nmi['deep'][z].values()) for z in tm]
    nmi_sc_std = [np.std(nmi['sc'][z].values()) for z in tm] 
    nmi_abp_std = [np.std(nmi['abp'][z].values()) for z in tm] 
    # get ccr for three algs
    ccr_deep_mean = [np.mean(ccr['deep'][z].values()) for z in tm]
    ccr_sc_mean = [np.mean(ccr['sc'][z].values()) for z in tm] 
    ccr_abp_mean = [np.mean(ccr['abp'][z].values()) for z in tm] 
    ccr_deep_std = [np.std(ccr['deep'][z].values()) for z in tm]
    ccr_sc_std = [np.std(ccr['sc'][z].values()) for z in tm] 
    ccr_abp_std = [np.std(ccr['abp'][z].values()) for z in tm] 
    # plot
    # x - ccr, o - nmi
    # b- - deep, r-- - sc, g-. - adp
    plt.figure(1)
    plt.errorbar(x_array, nmi_deep_mean, yerr=nmi_deep_std, fmt='bo-')
    plt.errorbar(x_array, nmi_sc_mean, yerr=nmi_sc_std, fmt='ro--')
    plt.errorbar(x_array, nmi_abp_mean, yerr=nmi_abp_std, fmt='go-.')
    
    plt.errorbar(x_array, ccr_deep_mean, yerr=ccr_deep_std, fmt='bx-')
    plt.errorbar(x_array, ccr_sc_mean, yerr=ccr_sc_std, fmt='rx--')
    plt.errorbar(x_array, ccr_abp_mean, yerr=ccr_abp_std, fmt='gx-.')
    
    plt.legend(['NMI-New', 'NMI-SC', 'NMI-ABP', 'CCR-New', 'CCR-SC', 'CCR-ABP'], loc=0)
    plt.xlabel('Performance as function of '+param)

    plt.show()
    return x_array

if __name__ == '__main__':
    # generating multiple graphs for the same parameter setting
    rand_tests = 5
    # setting storage space for results
#    nmi_arry = {}
#    ccr_arry = {}    
#    ars_arry = {}
    # parameter setting
#    c_array = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]  # alpha = clog(n)/n
#    c_array = [2.1,2.5,3.0]
    c_array = [4.0, 5.0]
    K_array = [2]  # number of communities
    N_array = [1000] # number of nodes
    lambda_array = [0.9] # B0 = lambda*I + (1-lambda)*ones(1,1)
    # scanning through parameters
    for c,K,N,lambda_n in itertools.product(c_array, K_array, N_array, lambda_array):
        print 'K:', K, 'N:', N, 'c:', c, 'lambda:', lambda_n
        model_sbm1 = SBM.SBM_param_init1(K, N, lambda_n, c)
        for rand in range(rand_tests):
            strsub1 = 'K'+str(K)+'N'+str(N)+'c'+str(c)+'la'+str(lambda_n)+'rd'+str(rand) # for saving results
            # simulate graph            
            G = SBM.SBM_simulate_fast(model_sbm1)
            ln, nodeslist = get_label_list(G)
            # algo1: proposed deepwalk algorithm
            model_w2v = SBM.SBM_learn_deepwalk_1(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
            X = model_w2v[nodeslist]
            k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
            k_means.fit(X)
            y_our = k_means.labels_ 
            nmi_arry, ccr_arry, ars_arry = summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_our, 'deep', 'c', c, rand)
            # algo2: spectral clustering 
            A = nx.to_scipy_sparse_matrix(G)
            print 'start spectral clustering'
            sc = SpectralClustering(n_clusters = K, affinity = 'precomputed', eigen_solver='arpack')
            sc.fit(A)
            y_sc= sc.labels_
            nmi_arry, ccr_arry, ars_arry = summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_sc, 'sc', 'c', c, rand)
            # algo3: belief propogation
            print 'start ABP algorithm'
            r = 3
            m, mp, lambda1 = SBM.abp_params(model_sbm1)
            y_abp = SBM.SBM_ABP(G, r, lambda1, m, mp)
            nmi_arry, ccr_arry, ars_arry = summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_abp, 'abp', 'c', c, rand)
    import pickle
    savename = 'exp101.pkl'
    res = [nmi_arry, ccr_arry, ars_arry]
    pickle.dump(res, open(savename, 'wb'), protocol=2)
#    
################### 
### use the following script to retrivle data and make new pltos
#import SBMmodels as SBM
#import networkx as nx
#from sklearn.cluster import KMeans
#from sklearn.cluster import SpectralClustering
#from sklearn import metrics
#import time
#import itertools
#import scipy
#import numpy as np
#import math
#import pickle
#
# ############ def plot_res_3_new(res, thr_wk, thr_hd,K):
#fname = 'exp1.pkl'
#res = pickle.load(open(fname, 'rb'))
#nmi_arry = res[0]
#ccr_arry = res[1] 
#ars_arry = res[2]
#
#thr_wk  = 0.395
#thr_pr  = 1.574
#thr_hd  = 4.28
#import matplotlib.pyplot as plt
#nmi = res[0]
#ccr = res[1]
#ars = res[2]
#tm = nmi['deep'].keys()
#param = tm[0].split('-')[0]
#x_array = [float(z.split('-')[1].strip()) for z in tm ]
#x_array = sorted(x_array)
#tm = [param + '- ' + str(v) for v in x_array]
## get nmi for three algs, mean and std
#nmi_deep_mean = [np.mean(nmi['deep'][z].values()) for z in tm]
#nmi_sc_mean = [np.mean(nmi['sc'][z].values()) for z in tm] 
#nmi_abp_mean = [np.mean(nmi['abp'][z].values()) for z in tm] 
#nmi_deep_std = [np.std(nmi['deep'][z].values()) for z in tm]
#nmi_sc_std = [np.std(nmi['sc'][z].values()) for z in tm] 
#nmi_abp_std = [np.std(nmi['abp'][z].values()) for z in tm] 
## get ccr for three algs
#ccr_deep_mean = [np.mean(ccr['deep'][z].values()) for z in tm]
#ccr_sc_mean = [np.mean(ccr['sc'][z].values()) for z in tm] 
#ccr_abp_mean = [np.mean(ccr['abp'][z].values()) for z in tm] 
#ccr_deep_std = [np.std(ccr['deep'][z].values()) for z in tm]
#ccr_sc_std = [np.std(ccr['sc'][z].values()) for z in tm] 
#ccr_abp_std = [np.std(ccr['abp'][z].values()) for z in tm] 
## plot
## x - ccr, o - nmi
## b- - deep, r-- - sc, g-. - adp
#plt.figure(1, figsize=(10, 6))
#
#plt.errorbar(x_array, nmi_deep_mean, yerr=nmi_deep_std, fmt='bo-.', markersize=8,linewidth= 1.5)
#plt.errorbar(x_array, nmi_sc_mean, yerr=nmi_sc_std, fmt='rs-.', markersize=8,linewidth= 1.5)
#plt.errorbar(x_array, nmi_abp_mean, yerr=nmi_abp_std, fmt='g<-.', markersize=8,linewidth= 1.5)
#
#plt.errorbar(x_array, ccr_deep_mean, yerr=ccr_deep_std, fmt='bo-', markersize=8,linewidth= 1.5)
#plt.errorbar(x_array, ccr_sc_mean, yerr=ccr_sc_std, fmt='rs-', markersize=8,linewidth= 1.5)
#plt.errorbar(x_array, ccr_abp_mean, yerr=ccr_abp_std, fmt='g<-', markersize=8,linewidth= 1.5)
#
#plt.legend(['NMI-New', 'NMI-SC', 'NMI-ABP', 'CCR-New', 'CCR-SC', 'CCR-ABP'], loc=0)
#plt.xlabel(param)
#plt.xlim(x_array[0]-0.1,x_array[-1]+0.1)
#plt.ylim(-0.05,1.05)
#plt.plot([thr_wk, thr_wk],[0,1],'k--', linewidth= 2.5)
#plt.plot([thr_pr, thr_pr], [0,1], 'k--', linewidth = 2.5)
#plt.plot([thr_hd, thr_hd], [0,1], 'k--', linewidth = 2.5)
##    plt.plot(x_array, [1./float(K)]*len(x_array), 'r--')
##    plt.plot(x_array, [1.0]*len(x_array), 'r--')
#plt.show()
#figurename = 'exp1'
#plt.savefig(figurename+'.eps',bbox_inches='tight', format='eps')
#plt.savefig(figurename+'.png',bbox_inches='tight', format='png')
##    return x_array
