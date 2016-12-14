# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:50:02 2016

@author: dingw
pipeline to:
1 learn embeddings from simulated data, 
2 apply standard K-means clustering
3 evaluate results/save results 
"""

import utils as SBM
import networkx as nx
import itertools
import numpy as np
import realworld_util as real

#%% 
if __name__ == '__main__':
    edgefname = '../data/politicalblog/Polblog_final_edges.txt'
    nodefname = '../data/politicalblog/Polblog_final_nodes.txt'
    G, model_s=real.parse_txt_data(edgefname, nodefname)
    K = model_s['K']
    ln = real.get_true_labels(G)

#%% 
    print 'vec algorithm'
    nodes4w2v = [str(x) for x in G.nodes()]
    rw_filename_vec = 'sentences_vec.txt'
    emb_filename_vec = 'emb_vec.txt'
    model_w2v = SBM.SBM_learn_deepwalk_1(G, 10, 60,  50, rw_filename_vec, emb_filename_vec, 8)
    X_w2v = model_w2v[nodes4w2v]
    y_our = SBM.clustering_embs(X_w2v, K)
    nmi_our, ccr_our, _ = SBM.cal_metrics_3(ln, y_our)
    print ccr_our
    print nmi_our

################ saveing results
###import pickle
###res = [nmi_our_arry, nmi_abp_arry, ccr_our_arry, ccr_abp_arry]
###pickle.dump(res, open('res_politicalblog.pkl', 'wb'))
############### visualization
#Y = SBM.SBM_visual_tsne(ln, X_lda)
#Ya0=[Y[i,0] for i in range(1222) if ln[i]==1]
#Ya1=[Y[i,1] for i in range(1222) if ln[i]==1]
#Yb0=[Y[i,0] for i in range(1222) if ln[i]==2]
#Yb1=[Y[i,1] for i in range(1222) if ln[i]==2]
#import matplotlib.pyplot as plt
#plt.figure(1, figsize=(8, 6))
##plt.scatter(Y[:,0], Y[:,1], 20, ln)
#plt.scatter(Ya0, Ya1, 20, 'b' , marker = 'o')
#plt.scatter(Yb0, Yb1, 20, 'r',  marker = 'x')
#plt.legend(['group 1', 'group 2'], loc=0)
#plt.show()
##figurename = 'blog_emb'
##plt.savefig(figurename+'.eps',bbox_inches='tight', format='eps')
##plt.savefig(figurename+'.png',bbox_inches='tight', format='png')
