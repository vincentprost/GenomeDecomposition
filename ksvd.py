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
from sklearn import linear_model
import multiprocessing
from itertools import product
import spams
from scipy.sparse import csc_matrix
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


hash_size = 27
n = 20
wd = ""
step = 2**18
cluster_cols = 5
clusters_nb = 1
cpu = 24
vectors = np.memmap(args.input + "/abundance_el1_27_cwn", dtype='float32', mode='r', shape=(n, 2**hash_size), order='F')
non_zeros = np.fromfile(args.input + "/non_zeros_el1_27", dtype='bool')

clusters_mm = np.memmap(args.output + "/kmer_clusters" + name2, dtype='int16', mode='w+', shape=(5, 2**hash_size), order='F')



nz = np.array(non_zeros)
nzi = np.sum(nz)
#del non_zeros

cols = 2**hash_size
vectors = np.array(vectors[:, nz])
matrix_size = int(nzi/2)



K = 200
D = np.zeros((n, K))
for i in range(K):
	ind = np.random.randint(nzi)
	D[:, i] = vectors[:, ind]







vectors = vectors[:, : matrix_size]








def rank1(C):
	l,c = np.shape(C)
	I = 0
	u = np.ones(l)
	w = np.ones(c)

	diff = np.inf
	while diff > 0:
		w = np.transpose(u).dot(C)
		w = w/np.linalg.norm(w)
		u = C.dot(w)
		u = u/np.linalg.norm(u)
		I_ = np.transpose(u).dot(C).dot(w)
		diff = np.abs(I - I_)
		I = I_
	return u, I



def ksvd(vectors, D):

	chunk_size = 2**18
	iter_nb = int(matrix_size / chunk_size) + 1
	K = 200
	D = np.zeros((n, K), dtype = np.float32)
	for i in range(K):
		ind = np.random.randint(matrix_size)
		D[:, i] = vectors[:, ind]

	for i in range(2):
		D_ = D.copy()
		print("start " + str(i))
		K = 200
		E_s = np.zeros((n, K))
		nz_inds = np.zeros((K, matrix_size), dtype = np.bool)

		param = {'lambda1' : 0.1, 'lambda2' : 0.1, 'numThreads' : 4,
           'pos' : False, 'mode' : 0}
		#a = spams.lasso(np.asfortranarray(vectors), D = np.asfortranarray(D),**param)
		a = spams.omp(np.asfortranarray(vectors), D = np.asfortranarray(D), eps = 0.0001, return_reg_path = False, numThreads = cpu)


		#nz_inds = a > 0
		Ds = csc_matrix(D)
		R = Ds.dot(a)

		indices = a.indices
		indptr = a.indptr
		data = a.data

		print(indptr)


		for c in np.arange(matrix_size - 1):
			nz_inds[indices[indptr[c]: indptr[c + 1]] ,c] = True


		for i in range(K):
			print(i)
			inds = nz_inds[i,:]
			if np.sum(inds) > 0:

				#E_i = vectors[:, inds]
				E_i = R[:, inds] +  np.outer(D[:,i], a.getrow(i).toarray()[:,inds])

				print("compute SVD")
				U, s, Vh = np.linalg.svd(E_i, full_matrices=False)
				print(Vh[0,:])
				D[:, i] = np.array(U)[:, 0]
				R[:, inds] = R[:, inds] + s[0] * np.outer(D[:,i], Vh[0,:])

		print(D)
		np.save(args.output + "/D" + name2, D )


	D = np.load(args.output + "/D" + name2 + ".npy")
	D = np.asfortranarray(D)

	print("write vectors")
	vectors = np.memmap(args.input + "/abundance_el1_27_cwn", dtype='float32', mode='r', shape=(n, 2**hash_size),  order = 'F')
	nz = np.array(non_zeros)
	nzi = np.sum(nz)

	vectors = np.array(vectors[:, nz])

	chunk_size = 2**18
	iter_nb = int(np.sum(nz) / chunk_size) + 1

	di = np.arange(2**hash_size, dtype = np.int32)[nz]



	for k in range(iter_nb):
		#for k in range(3):
		print(k)
		sup = min(nzi, (k + 1) * chunk_size)
		inds = np.arange(k * chunk_size, sup)
		v = vectors[:, k * chunk_size : sup]
		#a = lasso(v, D)
		a = spams.omp(np.asfortranarray(v), D = np.asfortranarray(D), eps = 0.0001, return_reg_path = False, numThreads = cpu).toarray()
		clusters = np.argsort(a, axis = 0)[-clusters_nb:][::-1]


		mask = a[clusters,  np.arange(len(inds))]
		mask = mask > 0
		clusters[~mask] = -1
		clusters_mm[:clusters_nb, di[inds]] = clusters + 1
		clusters_mm.flush()


	return clusters, vectors


"""
def lasso(x, D):
	param = {'lambda1' : 0.1, 'lambda2' : 0.1, 'numThreads' : cpu,
           'pos' : False, 'mode' : 0}
	a = spams.lasso(np.asfortranarray(x), D = D,**param).toarray()
	return a
"""

print(D)
clusters, vectors = ksvd(vectors, D)
