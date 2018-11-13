#!/usr/bin/env python
from __future__ import division


import sys, getopt
import glob, os
import gzip, re

import numpy as np
from scipy.spatial import distance
import random
import scipy.sparse as sp
import spams
import time
import multiprocessing
import argparse

name = "/export/home/vprost/workspace/LatentStrainAnalysis-master_light/matrices"
name2 = "_DL"

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

nzi = np.sum(non_zeros)

clusters = clusters_mm[0,:]
cols = 2**hash_size

di = np.arange(2**hash_size, dtype = np.int32)[non_zeros]
ii = np.cumsum(non_zeros).astype(np.int32) - 1


vectors = np.array(vectors[:, non_zeros])

print(vectors)

param = { 'mode' : 2, 'K' : K,
          'lambda1' : 0.1, 'lambda2' : 0.1, 'posAlpha' : True, 'numThreads' : 2, 'batchsize' : 10000,
          'iter' : 200, 'posD' : True}


lparam = { 'mode' : 2,
          'lambda1' : 0.1, 'lambda2' : 0.1, 'pos' : True, 'numThreads' : 2
}


D = np.asfortranarray(np.zeros((n, K), dtype = np.float32))



for i in range(K):
	ind = np.random.randint(nzi)
	D[:, i] = vectors[:, ind]


def dictionary_update(D, A, B):
	j = 0
	cols = np.shape(D)[1]
	diff = np.inf
	err = 0
	D_ = np.zeros((np.shape(D)))
	count = 0
	while diff > 10e-3:
		count += 1
		D_[:] = D[:]
		for j in range(cols):
			if A[j,j] > 0:
				u = (B[:,j] - D.dot(A[:,j]))/A[j,j] + D[:,j]
				D[:,j] = u/(np.max([np.linalg.norm(u), 1]))
				D[D[:,j] < 0,j] = 0

		diff = np.sum(np.abs(D - D_))
		if count > 200:
			print(" more than 200 iterations")
			break

	return D



def sparse_coding(i, D):

	A = np.zeros((K, K))
	B = np.zeros((n, K))


	chunk_size = 2000


	inds = np.arange(i, i + chunk_size)
	v = vectors[:, inds]
	a = spams.lasso(v, D = D,**lparam).toarray()

	for k in np.arange(len(inds)):
		A += np.outer(a[:,k],a[:,k])
		B += np.outer(v[:,k],a[:,k])

	return A, B



def write_part(i):
	means = np.load(args.output + "/D" + name2 + ".npy")

	if i == 0:
		print(np.shape(means))
		print(means)

	print(i)

	A = np.zeros((K, K))
	B = np.zeros((n, K))

	sup = min(nzi, (i + 1) * chunk_size)
	inds = np.arange(i * chunk_size, sup)
	#print("chunk_size " + str(chunk_size))

	v = vectors[:, inds]

	#D = distance.cdist(v.T, means.T, 'euclidean')
	D = distance.cdist(v.T, means.T, 'cosine')

	clusts = np.nanargmin(D, axis = 1)
	vals = D[np.arange(len(clusts)), clusts]
	ol = vals > np.inf

	clusters = np.zeros((len(inds),), dtype = np.int16)
	clusters[~ol] = clusts[~ol] + 1

	return inds, clusters


p = multiprocessing.Pool(3)

A = np.zeros((K, K), dtype = np.float64)
B = np.zeros((n, K), dtype = np.float64)



for k in np.arange(2, 100):

	print(str(k) + " ieme iteration")

	chunk_size = 2000

	"""
	iter_nb = 10

	offset = np.random.randint(nzi - chunk_size * iter_nb - 1)

	r = p.imap(sparse_coding, np.arange(offset, offset + iter_nb * chunk_size, chunk_size))

	A_ = np.zeros((K, K), dtype = np.float64)
	B_ = np.zeros((n, K), dtype = np.float64)
	for i in range(iter_nb):

		A__, B__ = r.next()
		A_ = A_ + A__
		B_ = B_ + B__

		#clusters[di[inds]] = c[:]
		#clusters.flush()
	beta = np.power(1 - 1./k, 15)


	"""


	offset = np.random.randint(nzi - chunk_size - 1)


	beta = np.power(1 - 1./k, 15)
	beta = 0.99
	A_, B_ = sparse_coding(offset, D)
	A = A + beta * A_
	B = B + beta * B_


	D = dictionary_update(D, A, B)

	#print(D, A, B)
	#print("D", np.sum(np.sum(D, 0) == 0))
	#print("A", np.sum(A[0:K, 0:K] == 0))


np.save(args.output + "/D" + name2, D)
print(D)

print("write clusters")
chunk_size = 2**18
iter_nb = int(nzi/chunk_size) + 1
print(iter_nb)

r = p.imap(write_part, range(0, iter_nb))

for i in range(iter_nb):
	inds, c = r.next()
	clusters[di[inds]] = c[:]
	clusters.flush()

# Save results
