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




class DictionaryLearner:

    def __init__(self, vectors, K = 200, cpu = 5, iter_nb = 100, chunk_size = 2000):
        self.K = 200
        self.cpu = 5
        self.vectors = vectors
        self.n = np.shape(vectors)[0]

        self.D = np.zeros((self.n, K))
        self.D = np.asfortranarray(np.zeros((self.n, K), dtype = np.float32))

        self.init_dictionary()


        self.chunk_size = chunk_size
        self.iter_nb = iter_nb
        self.lass_params = { 'mode' : 2,
                  'lambda1' : 0.1, 'lambda2' : 0.1, 'pos' : True, 'numThreads' : 2
        }

    def __str__(self):

        return "DictionaryLearner"

    def sparse_coding_lasso(self, i):
        A = np.zeros((self.K, self.K))
        B = np.zeros((self.n, self.K))
        self.chunk_size = 2000
        inds = np.arange(i, i + self.chunk_size)
        v = self.vectors[:, inds]
        a = spams.lasso(v, D = self.D,**self.lass_params).toarray()


        for k in np.arange(len(inds)):
        	A += np.outer(a[:,k],a[:,k])
        	B += np.outer(v[:,k],a[:,k])

        return A, B



    def dictionary_update(self, A, B):
        j = 0
        diff = np.inf
        err = 0
        D_ = np.zeros((self.n, self.K))
        D = self.D
        count = 0
        while diff > 10e-3:
            count += 1
            D_[:] = D[:]
            for j in range(self.K):
                if A[j,j] > 0:
                	u = (B[:,j] - D.dot(A[:,j]))/A[j,j] + D[:,j]
                	D[:,j] = u/(np.max([np.linalg.norm(u), 1]))
                	D[D[:,j] < 0,j] = 0
            diff = np.sum(np.abs(D - D_))
            if count > 200:
            	print(" more than 200 iterations")
            	break
        return D


    def dictionary_learning(self):

        A = np.zeros((self.K, self.K), dtype = np.float64)
        B = np.zeros((self.n, self.K), dtype = np.float64)

        for k in np.arange(2, self.iter_nb):
            offset = np.random.randint(np.shape(self.vectors)[1] - self.chunk_size - 1)
            beta = 0.99
            A_, B_ = self.sparse_coding_lasso(offset)
            A = A + beta * A_
            B = B + beta * B_
            D = self.dictionary_update(A, B)
            self.D = D
        return self.D


    def init_dictionary(self):
        for i in range(self.K):
        	ind = np.random.randint(np.shape(self.vectors)[1])
        	self.D[:, i] = self.vectors[:, ind]
