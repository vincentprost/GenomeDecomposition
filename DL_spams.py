#!/usr/bin/env python
import sys, getopt
import glob, os
import gzip, re

import numpy as np
from scipy.spatial import distance
import random
from collections import defaultdict
import scipy.sparse as sp
import spams
import time
import multiprocessing
import argparse




name = "/export/home/vprost/workspace/LatentStrainAnalysis-master_light/matrices"
name2 = "_NMF_spams"

parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('-i', '--input',  type=str,
                    help='input folder')

parser.add_argument('-o', '--output', type=str, default="matrices",
                    help='output folder')

args = parser.parse_args()


if not os.path.exists(args.output):
    os.makedirs(args.output)



print("matrix "  + args.input)

hash_size = 27
n = 20
wd = ""
step = 2**18
cluster_cols = 5
clusters_nb = 1
cpu = 5
K = 200
chunk_size = 2**18
random.seed(0)



vectors = np.memmap(args.input + "/abundance_el1_27_cwn", dtype='float32', mode='r', shape=(n, 2**hash_size), order='F')
non_zeros = np.fromfile(args.input + "/non_zeros_el1_27", dtype='bool')

clusters_mm = np.memmap(args.output + "/kmer_clusters" + name2, dtype='int16', mode='w+', shape=(5, 2**hash_size), order='F')


X = np.array(vectors[:, non_zeros])
X = np.asfortranarray(X)


print(X)
print("input ok")


param = { 'K' : 200,
          'lambda1' : 0.1, 'lambda2' : 0.1, 'posAlpha' : False, 'numThreads' : cpu, 'batchsize' : 10000,
          'iter' : 10, 'posD' : True, 'mode' : 0}



D = spams.trainDL(X, **param)

np.save(args.output + "/DL" + name2, D)
print(D)



lparam = { 'mode' : 2,
          'lambda1' : 0.1, 'lambda2' : 0.1, 'pos' : True, 'numThreads' : 1
}


print(lparam)
D = np.load(args.output + "/DL" + name2 + ".npy")
print(D)


#norms = np.zeros((2**hash_size), dtype = np.float32)

def write_part(i):
    print("start " + str(i))
    inds = np.arange(i * chunk_size, min(nb_nz, (i + 1) * chunk_size))


    nz_inds = gr[non_zeros][inds]
    x = X[:, inds]
    a = spams.lasso(x, D = D, **lparam).toarray()
    clusters = np.argsort(a, axis = 0)[-clusters_nb:][::-1]

    mask = a[clusters,  np.arange(len(inds))]
    mask = mask > 0

    clusters[~mask] = -1

    alpha[:clusters_nb, nz_inds] = clusters + 1
    alpha.flush()


    print(str(i) + " ok !")
    return 0


chunk_size = 2**17
gr = np.arange(2**hash_size)
nb_nz = np.sum(non_zeros)

alpha = np.memmap(args.output + "/kmer_clusters" + name2, dtype='int16', mode='w+', shape=(5, 2**hash_size), order = 'F')

print(int(np.sum(non_zeros)/chunk_size) + 1)
print(int(nb_nz/chunk_size) + 1)

p = multiprocessing.Pool(cpu)
r = p.imap(write_part, range(0, int(nb_nz/chunk_size) + 1 ))


for k in range(0, int(nb_nz/chunk_size) + 1 ):
	r.next()

del alpha
