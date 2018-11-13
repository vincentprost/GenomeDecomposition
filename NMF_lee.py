#!/usr/bin/env python
from __future__ import division


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
name2 = "_NMF_lee"


parser = argparse.ArgumentParser(description='Optional app description')


parser.add_argument('-i', '--input',  type=str,
                    help='input folder')

parser.add_argument('-o', '--output', type=str, default="matrices",
                    help='output folder')

args = parser.parse_args()


if not os.path.exists(args.output):
    os.makedirs(args.output)



hash_size = 27
n = 20
wd = ""
step = 2**18
cluster_cols = 5
clusters_nb = 1
cpu = 20

K = 200
chunk_size = 2**18

vectors = np.memmap(args.input + '/kmer_counts_el1_27_kl', dtype='float32', mode='r', shape=(n, 2**hash_size), order='F')
clusters_mm = np.memmap(args.output + "/kmer_clusters"  + name2 , dtype='int16', mode='w+', shape=(5, 2**hash_size), order='F')
non_zeros = np.memmap(args.input +  '/non_zeros_el1_27', dtype='bool', mode='r', shape= (2**27))



nz = np.array(non_zeros)
nzi = np.sum(nz)

clusters = clusters_mm[0,:]
cols = 2**hash_size

di = np.arange(2**hash_size, dtype = np.int32)[nz]
ii = np.cumsum(nz).astype(np.int32) - 1


chunk_size = int(nzi/4)

vectors = np.array(vectors[:, nz])

print(vectors)

K = 200

D = np.asfortranarray(np.zeros((n, K), dtype = np.float32))
for i in range(K):
	ind = np.random.randint(chunk_size)
	D[:, i] = vectors[:, ind]



alpha = np.random.rand(K, chunk_size).astype(np.float32)
#alpha = np.ones((K, chunk_size)) / K


def lee_kl_D(X, D, alpha):
	alpha_ = alpha.T / np.sum(alpha, axis = 1)
	return (np.divide(X, D  @  alpha) @ alpha_) * D

def lee_kl_alpha(X, D, alpha):
	D_ = (D / np.sum(D, axis = 0)).T
	return (D_ @ np.divide(X, D  @  alpha) ) * alpha

def nmf_lee_and_seung(X, dim = -1):
	if dim == -1:
		dim = np.shape(X)[1]

	m, n = np.shape(X)

	W = np.random.rand(m, dim)
	H = np.random.rand(dim, n)
	error = np.inf


	while True:
		W = np.nan_to_num(lee_W(X, W, H))
		H = np.nan_to_num(lee_H(X, W, H))

		print("error :")
		error_ = np.linalg.norm(X - W.dot(H))
		print(error_)
		if np.abs(error_ - error) < 0.0001:
			break
		error = error_
	return W, H



#D = spams.trainDL(X, **param)
#D = spams.structTrainDL(X, **param)
#D = spams.trainDL(X, **param)


def write_part_kl(i):
	print(i)
	D = np.load("matrices/D" + name2 + ".npy")


	sup = min(nzi, (i + 1) * chunk_size)
	inds = np.arange(i * chunk_size, sup)

	v = vectors[:, inds]
	lD = np.log(D)
	clusters = np.zeros((len(inds),), dtype = np.int16)

	for k in np.arange(len(inds)):

		kld = v[:,k].dot(lD)
		clust = np.nanargmax(kld)
		clusters[k] = clust + 1

	return inds, clusters


X = vectors[:,:chunk_size]
D_ = D
diff = np.inf
k = 0



while diff > 1e-4:
	k += 1
	print(str(k ) + " eme iteration..")
	alpha = lee_kl_alpha(X, D, alpha)
	D = lee_kl_D(X, D, alpha)
	print(D)
	diff = np.max(np.abs(D - D_))
	print(diff)
	np.save(args.output + "D" + name2, D)
	D_ = D


D = D / np.sum(D, axis = 0)

np.save(args.output + "/D" + name2, D)
print(D)


p = multiprocessing.Pool(3)
print("write clusters")
chunk_size = 2**16
iter_nb = int(nzi/chunk_size) + 1


r = p.imap(write_part_kl, range(iter_nb))

for i in range(iter_nb):
	inds, c = r.next()
	clusters[di[inds]] = c[:]
	clusters.flush()

exit()
