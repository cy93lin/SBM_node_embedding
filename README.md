## Word embedding for Community Detection in Graph

This repository contains all the functions and scripts for getting word2vec embeddings for community detection on graphs. For details please refer to [the following paper](https://arxiv.org/abs/1611.03028):

W Ding, C Lin, P Ishwar, "Node Embedding via Word Embedding for Network Community Discovery", in arXiv preprint arXiv:1611.03028, 2016. 

A shorter version of this is in IEEE ICASSP conference (2017). 


### Dependencies:

The functions in this repository requires only basic python packages (scipy, numpy, sklearn, and gensim). No additional installation is required. 

### Usage:

The functions to create embeddings and communities are in VEClib.py and utils.py. 

Please check the example code in /example folder to see how these functions can be used for community detection. 

We provide two example scripts in /example folder to demo how to use our code for community detection for simulated graphs, as well as real word graphs. 

*NOTE*: /all_script folder has all the scripts for re-creating the plots in the abovementioned paper where we demonstrates the empirical performance around the information theory limits of Stochastic Block Models. 