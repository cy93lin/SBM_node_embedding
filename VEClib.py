"""
This file contains all the functions for node embedding

## from graph to embedding:
- SBM_learn_deepwalk_1(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
write paths in to file and then optimize
- SBM_learn_deepwalk_2(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
write paths in to file and then optimize using c-code implementation
- SBM_learn_deepwalk_3(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
save paths in memory and then optimize


## from path to embedding:
- SBM_learn_fromcorpus_1(rw_filename, emb_dim, winsize,  emb_filename):
read already existed paths and optimize
- SBM_learn_fromcorpus_2(rw_filename, emb_dim, winsize,  emb_filename):
read already existed paths and optimize


### also, all the functions for ABP algorithm in Abbe et al., 2015, 2016
- SBM_ABP(G, r, lambda1, m, mp)
abp for two communities 
- multi_abp(G,r,lambda1, m, mp, dim, K)
abp for mutiple communities
- abp_params(md)
get algorithm parameters for ABP



"""
import math
import numpy as np
import numpy.random as npr
import networkx as nx
import gensim.models.word2vec as w2v
from sklearn import metrics
import scipy

###########################
# helper functions
def alias_setup(probs):
    """
    This function is to help draw random samples from discrete distribution with specific weights,
    the code were adapted from the following source:
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/    
    
    arguments:
    probs: the discrete probability
    return:
    J, q: temporary lists to assist drawing random samples
    """    
    K       = len(probs)
    q       = np.zeros(K)
    J       = np.zeros(K, dtype=np.int)
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
 
        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
 
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q
    
def alias_draw(J, q):
    """
    This function is to help draw random samples from discrete distribution with specific weights,
    the code were adapted from the following source:
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/    
    
    arguments:
    J, q: generated from alias_setup(prob)
    return:
    a random number ranging from 0 to len(prob)
    """
    K  = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand()*K))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]
##########################
        
    
def SBM_savemat(G, edgefilename, nodefilename):
    """
    writting down G graph by its nodes attributes and 
    """
    nx.write_edgelist(G, edgefilename, data = False)
    nodeslist = G.nodes()
    with open(nodefilename, 'w') as fwrite:
        for key in nodeslist:
            fwrite.write(str(key)+' '+str(G.node[key]['community'])+'\n')
    return 1

#############################
def build_node_alias(G):
    """
    build dictionary S that is easier to generate random walks on G
    G is networkx objective
    return: nodes_rw with J, q for each node created using alias_draw functions
    """
    nodes = G.nodes()
    nodes_rw = {}
    for nd in nodes:
        d = G[nd]
        entry = {}
        entry['names'] = [key for key in d]
 #       weights = [math.exp(d[key]['weight']) for key in d]
        weights = [d[key]['weight'] for key in d]        
        sumw = sum(weights)
        entry['weights'] = [i/sumw for i in weights]
        J,q = alias_setup(entry['weights'])
        entry['J'] = J
        entry['q'] = q
        nodes_rw[nd] = entry
    return nodes_rw
    
def create_rand_works(S, num_paths, length_path, filename):
    """
    S = from the build_node_alias
    filename - where to write results
    using exp(rating) as edge weight
    """
    fwrite = open(filename,'w')
    nodes = S.keys()
    for nd in nodes:
        for i in range(num_paths):
            walk = [nd] # start as nd
            for j in range(length_path):
                cur = walk[-1]
                next_nds = S[cur]['names']
                if len(next_nds)<1:
                    break
                else:
                    J = S[cur]['J']
                    q = S[cur]['q']
                    rd = alias_draw(J,q)
                    nextnd = next_nds[rd]
                    walk.append(nextnd)   
            walk = [str(x) for x in walk]
            fwrite.write(" ".join(walk) + '\n')
    fwrite.close()
    return 1
    
def create_rand_works_inmem(S, num_paths, length_path):
    """
    S = from the build_node_alias
    filename - where to write results
    using exp(rating) as edge weight
    """
    sentence = []
    nodes = S.keys()
    for nd in nodes:
        for i in range(num_paths):
            walk = [nd] # start as nd
            for j in range(length_path):
                cur = walk[-1]
                next_nds = S[cur]['names']
                if len(next_nds)<1:
                    break
                else:
                    J = S[cur]['J']
                    q = S[cur]['q']
                    rd = alias_draw(J,q)
                    nextnd = next_nds[rd]
                    walk.append(nextnd)   
            walk = [str(x) for x in walk]
            sentence.append(walk)
    return sentence
################################  
def SBM_learn_writecorpus1(G, num_paths, length_path, rw_filename):
    print '1 building alias auluxy functions'
    S = build_node_alias(G)    
    print '2 creating random walks'
    create_rand_works(S, num_paths, length_path, rw_filename)
    return 1
def SBM_learn_fromcorpus_1(rw_filename, emb_dim, winsize,  emb_filename):
    print '3 learning word2vec models'
    sentence = w2v.LineSentence(rw_filename) 
    model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=winsize, \
                             min_count=0, sg = 1, negative = 5, sample = 1e-1, workers=4, iter = 3)
    return model_w2v

def SBM_learn_fromcorpus_2(rw_filename, emb_dim, winsize,  emb_filename):
    import os
    print '3 learning word2vec models using C code'
    comman = './word2vec -train '+rw_filename+' -output '+emb_filename+' -size '+str(emb_dim)+' -window '+str(winsize)+' -negative 5 -cbow 0 -min-count 0 -iter 5 -sample 1e-1'
    os.system(comman)
    model_w2v = w2v.Word2Vec.load_word2vec_format(emb_filename, binary=False)
    return model_w2v

    
def SBM_learn_deepwalk_1(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize):
    """
    learning SBM model through deepwalk, using gensim package
    File I/O involved:
    first write all the randwalks on disc, then read to learn word2vec
    speed is relatively slow, but scales well to very large dataset
    Inputs:
    G: graph
    num_paths: number of random walks starting from each node
    length_path: length of each random walk
    rw_filename: file name to store the created corpus of sentences from graph G
    emb_filename: file name to store the learned embeddings of the nodes
    emb_dim: the dimensionality of the embeddings
    """
    print '1 building alias auluxy functions'
    S = build_node_alias(G)    
    print '2 creating random walks'
    create_rand_works(S, num_paths, length_path, rw_filename)
    print '3 learning word2vec models'
    sentence = w2v.LineSentence(rw_filename)
    model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=winsize, min_count=0, sg = 1, negative = 5, sample = 1e-1, workers=5, iter=3)
#    print '4 saving learned embeddings'    
#    model_w2v.save_word2vec_format(emb_filename)
    return model_w2v    

def SBM_learn_deepwalk_2(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize):
    """
    learning SBM model through deepwalk
    Using the word2vec C implementation, some computational tricks involved. 
    Writing random walks on file
    can scale to large dataset well
    Inputs:
    G: graph
    num_paths: number of random walks starting from each node
    length_path: length of each random walk
    rw_filename: file name to store the created corpus of sentences from graph G
    emb_filename: file name to store the learned embeddings of the nodes
    emb_dim: the dimensionality of the embeddings
    """
    import os
    print '1 building alias auluxy functions'
    S = build_node_alias(G)    
    print '2 creating random walks'
    create_rand_works(S, num_paths, length_path, rw_filename)
    print '3 learning word2vec models using C code'
    comman = './word2vec -train '+rw_filename+' -output '+emb_filename+' -size '+str(emb_dim)+' -window '+str(winsize)+' -negative 5 -cbow 0 -min-count 0 -iter 5 -sample 1e-1'
    os.system(comman)
    model_w2v = w2v.Word2Vec.load_word2vec_format(emb_filename, binary=False)
    print '4 saving learned embeddings'    
    model_w2v.save_word2vec_format(emb_filename)
    return model_w2v 
    
def SBM_learn_deepwalk_3(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize):
    """
    learning SBM model through deepwalk, using gensim package
    saving all the sentences in memory, saves a lot of file I/O time
    can achieve 3x speed up compare to File I/O approach
    can not scale to very large networks
    
    Inputs:
    G: graph
    num_paths: number of random walks starting from each node
    length_path: length of each random walk
    rw_filename: file name to store the created corpus of sentences from graph G
    emb_filename: file name to store the learned embeddings of the nodes
    emb_dim: the dimensionality of the embeddings
    """
    print '1 building alias auluxy functions'
    S = build_node_alias(G)    
    print '2 creating random walks'
    sentence = create_rand_works_inmem(S, num_paths, length_path)
    print '3 learning word2vec models'
#    sentence = w2v.LineSentence(rw_filename)
    model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=winsize, min_count=0, sg = 1, negative = 5, sample = 1e-1, workers=4)
#    print '4 saving learned embeddings'    
#    model_w2v.save_word2vec_format(emb_filename)
    return model_w2v    

    
def SBM_visual_tsne(labels, X):    
    import tsne
    import pylab as Plot
    Y=tsne.tsne(X, 2)
    Plot.figure()
    Plot.scatter(Y[:,0], Y[:,1], 20, labels);
    Plot.show();
    return Y

def check_cycle(G,u,v,r):
    # u,v is path of cycle <=r <=> u,v has shortest path<=r-1 after removing (u,v)
    if r==1:
        ispartcycle = False
    else:
        # save a copy of current edge
        edgetmp = {}
        entry = G[u][v]
        for key in entry:
            edgetmp[key]= entry[key]
        # delete current edge from graph
        G.remove_edge(u,v)
        # check if shortest path <= r-1
        if nx.has_path(G,u,v):
            z = nx.shortest_path(G, u, v)
            if len(z)-1 <= r-1:
                ispartcycle  = True
            else:
                ispartcycle  = False
        else:
            ispartcycle = False
        # add edge back to graph
        G.add_edge(u,v)
        for key in edgetmp:
            G[u][v][key] = edgetmp[key]
    # now return 
    if ispartcycle:
        return ispartcycle, z
    else:
        return ispartcycle, []
        
def SBM_ABP(G, r, lambda1, m, mp):
    # step 1: initialize Y(v,v')(0) 
    t = 1
    elist = G.edges()
    cstr = 'cycles<='+str(r)
    Y = {}
    for e in elist:
        v=e[0]
        vp=e[1]
        erev = (vp,v)
        Y[e] = {}
        Y[e][t] = np.random.normal(0,1)  # y(v,v')(1)
        Y[erev] = {}
        Y[erev][t] = np.random.normal(0,1) # y(v',v)(1)
    # step 2: check if v,v' is part of a cycle <=r
    for e in elist:
        v = e[0]
        vp = e[1]
        erev = (vp,v)
        iscycle,z = check_cycle(G,v,vp, r)
        Y[e][cstr] = {}
        Y[e][cstr]['bin'] = iscycle
        if iscycle:
            Y[e][cstr]['path'] = z
        Y[erev][cstr] = {}
        Y[erev][cstr]['bin'] = iscycle
        if iscycle:
            Y[erev][cstr]['path'] = z
    # step 3: iterations to calculate y(v,v')(t), 1<t<=m
    elist = Y.keys()
    for t in range(2,m+1):
        for e in elist:
            v = e[0]
            vp = e[1]
            wts = [Y[(vp, vpp)][t-1] for vpp in G[vp] if vpp not in [v]]
            if Y[e][cstr]['bin'] is False:
                # v,v' not in a cycle of length <=r, use eq a) to get y(v,v')(t)
                Y[e][t] = sum(wts)
            else:  
                # v v' in a cycle of length r' (where r'<=r)
                z = Y[e][cstr]['path']
                rp = len(z)
                # get the adj to v
                if z[0] == v:
                    vppp = z[1]
                else:
                    vppp = z[-2]
                # if cycle r'==t, use eq c) to get y(v,v')(t)
                if rp == t:
                    Y[e][t] = sum(wts) - float(len(wts))*Y[(vppp,v)][1]
                else:
                    mu = t- rp
                    if mu<1:
                        Y[e][t] = sum(wts)
                    else:
                        wts2 = [Y[(v,vpp)][mu] for vpp in G[v] if vpp not in [vp] and vpp not in [vppp] ]
                        Y[e][t] = sum(wts) - sum(wts2)
    # step 4: get the Y matrix
    Ymat = {}
    nds = G.nodes()
    for v in nds:
        Ymat[v] = {}
        for t in range(1,m+1):
            yts = [Y[(v, s)][t] for s in G[v]]
            Ymat[v][t] = sum(yts)
    # step 5: calculate y', and clustering
    M = np.diag([-1*lambda1]*(m-1),k=1)+np.eye(m)
    em = np.zeros((m,1))
    em[m-1]=1
    for j in range(1,mp):
        em = np.dot(M, em)
    labels_est = []
    for v in nds:
        tmp =[Ymat[v][t+1]*em[t][0] for t in range(0,m)]
        Ymat[v]['yp']= sum(tmp)
        if Ymat[v]['yp']>0.0:
            labels_est.append(1)
        else:
            labels_est.append(0)    
    return labels_est

def multi_abp(G,r,lambda1, m, mp, dim, K):
    from sklearn.cluster import KMeans
    N = len(G.nodes())
    mt = np.zeros((N, dim))
    for k in range(dim):
        print 'k-th iter:', k
        y_abp = SBM_ABP(G, r, lambda1, m, mp)
        mt[:,k] = y_abp
    k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
    k_means.fit(mt)
    y = k_means.labels_
    return y  
    
    
    
def parse_txt_data(edgename, nodename):
    """
    additional helper fucntion for parsing graph data
    in txt files by Christy
    edgename: edge files,  each line: u,v  (node ids of a edge)
    nodename: node file, each line: nodeid, node-labels
    """
    model_params = {}
    G = nx.Graph()
    # add edges
    f = open(edgename, 'r')
    for l in f:
        c = l.split()
        G.add_edge(c[0],c[1], weight = 1.0)
    f.close()
    # add nodes
    g = open(nodename, 'r')
    for l in g:
        c = l.split()
        G.node[c[0]]['community'] = c[1]
    g.close()
    # get overall graph info    
    nds = G.nodes()
    model_params['N'] = len(nds)
    labels = [G.node[i]['community'] for i in nds]
    uK = set(labels)
    model_params['K'] = len(uK)
    return G, model_params

def save_clusters_in_parallel(y, y_est,filename):
    """
    helper function to save the learned clustering results
    y - ground truth labels
    y_est - learned labels
    filename - to save results
    """
    f=open(filename, 'w')
    for i in range(len(y)):
        f.write(str(y[i]) + ','+str(y_est[i])+'\n')
    f.close
    return 1
    
    
def cal_metrics(labels, y_est_full):
    N = len(labels)
    acc_nmi = metrics.normalized_mutual_info_score(labels, y_est_full)
#    return acc_nmi
    Conf = metrics.confusion_matrix(labels, y_est_full)
    r,c = scipy.optimize.linear_sum_assignment(-1*Conf)
    acc_ccr = float(Conf[r,c].sum())/float(N)
    return acc_nmi, acc_ccr
    
def cal_modularity(G, nodelist, y):
    m = G.size()
    m = float(m)
    Q = 0.0
    n = len(nodelist)
    k=[]
    for e in nodelist:
        k.append(float(len(G[e])))
    for i in range(n):
        for j in range(i+1,n):
            if y[i] == y[j]:
                if G.has_edge(nodelist[i], nodelist[j]):
                    A = 1.0 - k[i]*k[j]/m
                else:
                    A = -1*k[i]*k[j]/m
                Q +=A
    return 2*Q/m
        

def SBM_SNR(model):
    """
    help to define the SNR and lambda1
    """
    Q = model['B0']*model['alpha']*float(model['N'])
    P = np.diag(model['a'])
    Z = np.dot(P,Q)
    u,v = np.linalg.eig(Z)
    ua = sorted(u, reverse = True) 
    print 'lambda1:', ua[0]
    print 'lambda2:', ua[1]
    SNR = ua[1]*ua[1]/ua[0]
    return SNR,ua[0],ua[1]

def get_true_labels(G):
    """
    only for Christy's format
    """
    nodeslist = G.nodes()
    labels = [G.node[i]['community'] for i in nodeslist]
    ln = [int(t) for t in labels]
    return ln
    
    
def abp_params(md):
    """
    this is only good for simulated data
    """
    snr, lambda1, lambda2 = SBM_SNR(md)
    n = md['N']
    m = 2.0*math.log(float(n))/math.log(snr)
    m = int(math.ceil(m)) + 1
    if m < 0:
        m = 2
    mp = m*math.log(lambda1*lambda1/lambda2/lambda2)/math.log(float(n))
    mp = int(math.ceil(mp)) +1
    if mp<0:
        mp = 2
    return m, mp, lambda1

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