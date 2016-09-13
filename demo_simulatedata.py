"""
This is an example script to show how to use the functions in VEClib.py and SBMlib.py


"""
import SBMlib as SBM
import VEClib as algs
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import itertools
import scipy
import numpy as np
   
# MACRO parameter setting
rw_filename = 'sentences.txt'
emb_filename = 'emb.txt'
num_paths =10
length_path = 60
emb_dim = 64
winsize = 8


if __name__ == '__main__':
    # generating multiple graphs for the same parameter setting
    rand_tests = 5
    # setting storage space for results
    nmi_arry = {}
    ccr_arry = {}    
    ars_arry = {}
    # parameter setting
    c_array = [2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]  # alpha = clog(n)/n
    K_array = [2]  # number of communities
    N_array = [10000] # number of nodes
    lambda_array = [0.9] # B0 = lambda*I + (1-lambda)*ones(1,1)
    # scanning through parameters
    for c,K,N,lambda_n in itertools.product(c_array, K_array, N_array, lambda_array):
        print 'K:', K, 'N:', N, 'c:', c, 'lambda:', lambda_n
        model_sbm1 = SBM.SBM_param_init_n(K, N, lambda_n, c)
        for rand in range(rand_tests):
            strsub1 = 'K'+str(K)+'N'+str(N)+'c'+str(c)+'la'+str(lambda_n)+'rd'+str(rand) # for saving results
            # simulate graph            
            G = SBM.SBM_simulate_fast(model_sbm1)
            ln, nodeslist = algs.get_label_list(G)
            # algo1: proposed deepwalk algorithm
            model_w2v = algs.SBM_learn_deepwalk_1(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
            X = model_w2v[nodeslist]
            k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
            k_means.fit(X)
            y_our = k_means.labels_ 
            nmi_arry, ccr_arry, ars_arry = algs.summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_our, 'deep', 'c', c, rand)
            # algo2: spectral clustering 
            A = nx.to_scipy_sparse_matrix(G)
            print 'start spectral clustering'
            sc = SpectralClustering(n_clusters = K, affinity = 'precomputed', eigen_solver='arpack')
            sc.fit(A)
            y_sc= sc.labels_
            nmi_arry, ccr_arry, ars_arry = algs.summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_sc, 'sc', 'c', c, rand)
            # algo3: belief propogation
            print 'start ABP algorithm'
            r = 3
            m, mp, lambda1 = algs.abp_params(model_sbm1)
            y_abp = algs.SBM_ABP(G, r, lambda1, m, mp)
            nmi_arry, ccr_arry, ars_arry = algs.summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_abp, 'abp', 'c', c, rand)
    import pickle
    savename = 'exp102.pkl'
    res = [nmi_arry, ccr_arry, ars_arry]
    pickle.dump(res, open(savename, 'wb'), protocol=2)
