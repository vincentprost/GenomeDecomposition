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
import pickle4reducer
import multiprocessing as mp
ctx = mp.get_context()
ctx.reducer = pickle4reducer.Pickle4Reducer()




class DictionaryLearner:

    def __init__(self, K = 200, cpu = 5, iter_nb = 100, chunk_size = 2000):
        self.K = K
        self.cpu = 5
        #self.vectors = vectors
        #self.n = np.shape(vectors)[0]

        #self.D = np.zeros((self.n, K))
        #self.D = np.asfortranarray(np.zeros((self.n, K), dtype = np.float32))

        #self.init_dictionary()


        self.chunk_size = chunk_size
        self.iter_nb = iter_nb
        self.lass_params = { 'mode' : 2,
                  'lambda1' : 0.1, 'lambda2' : 0.1, 'pos' : True, 'numThreads' : 2
        }

    def __str__(self):

        return "DictionaryLearner"

    def set_dictionary(self, D):
        self.D = D


    def sparse_coding_lasso(self, i, vectors):
        A = np.zeros((self.K, self.K))
        B = np.zeros((self.n, self.K))
        self.chunk_size = 2000
        inds = np.arange(i, i + self.chunk_size)
        v = vectors[:, inds]
        a = spams.lasso(v, D = self.D,**self.lass_params).toarray()


        for k in np.arange(len(inds)):
        	A += np.outer(a[:,k],a[:,k])
        	B += np.outer(v[:,k],a[:,k])

        return A, B



    def update_dictionary(self, A, B):
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


    def learn_dictionary(self, vectors):
        self.n = np.shape(vectors)[0]
        self.init_dictionary(vectors)
        A = np.zeros((self.K, self.K), dtype = np.float64)
        B = np.zeros((self.n, self.K), dtype = np.float64)

        for k in np.arange(2, self.iter_nb):
            offset = np.random.randint(np.shape(vectors)[1] - self.chunk_size - 1)
            beta = 0.99
            A_, B_ = self.sparse_coding_lasso(offset, vectors)
            A = A + beta * A_
            B = B + beta * B_
            D = self.update_dictionary(A, B)
            self.D = D
        return self.D




    def init_dictionary(self, vectors):

        self.D = np.asfortranarray(np.zeros((self.n, self.K), dtype = np.float32))
        for i in range(self.K):
        	ind = np.random.randint(np.shape(vectors)[1])
        	self.D[:, i] = vectors[:, ind]



    def find_best_clusters(i, vectors):
        #sup = min(np.shape(vectors)[1], (i + 1) * self.chunk_size)
        #inds = np.arange(i * self.chunk_size, sup)
        #v = vectors[:, inds]

        #D = distance.cdist(v.T, means.T, 'euclidean')
        dd = distance.cdist(vectors.T, self.D.T, 'cosine')

        clusts = np.nanargmin(dd, axis = 1)

        vals = dd[np.arange(len(clusts)), clusts]
        ol = vals > np.inf

        clusters = np.zeros((np.shape(vectors)[1],), dtype = np.int16)
        clusters[~ol] = clusts[~ol] + 1

        return clusters





class ClusterWriter:

    #def __init__(self, input_data, output_directory, non_zero_columns = None, nrow = 0, ncols = 0,  dictionary_learner = None,
    #            data_type = 'float32', param = None):
    def __init__(self, param = None):

        #self.input_data = input_data
        #self.dictionary_learner = dictionary_learner
        #self.output_directory = output_directory
        #self.data_type = data_type
        self.param = param
        #self.p = mp.Pool(1)

    def read_data(self, input_data, nrows = 0, ncols = 0, data_type = 'float32', non_zero_columns = None, order='F'):

        error_message = "error, could not read data"
        matrix_loaded = False
        if non_zero_columns != None:
            try:
                non_zero_columns = np.load(non_zero_columns)
            except:
                non_zero_columns = np.fromfile(non_zero_columns, dtype = 'bool')
            if non_zero_columns is None:
                return error_message

        try:
            vectors = np.load(input_data)
            if non_zero_columns is not None:
                vectors = vectors[:, non_zero_columns]
            matrix_loaded = True

        except:
            pass


        if not matrix_loaded:
            vectors = np.memmap(input_data, dtype = data_type, mode='r', shape=(nrows, ncols), order = order)
            if non_zero_columns is not None:
                vectors = np.array(vectors[:, non_zero_columns])
            else:
                vectors = np.array(vectors)
            matrix_loaded = True


        if not matrix_loaded:
            return error_message
        else:
            self.vectors = vectors

        if non_zero_columns is None:
            self.non_zero_columns = np.ones((np.shape(self.vectors)[0]), dtype = 'bool')
        else:
            self.non_zero_columns = non_zero_columns

    def write_part(self, i):
        print("start ok ")

        sup = min(np.shape(self.vectors)[1], (i + 1) * self.chunk_size)
        inds = np.arange(i * self.chunk_size, sup)

        v = self.vectors[:, inds]

        clusters = self.dictionary_learner.find_best_clusters(i, v)
        return inds, clusters


    def write_clusters(self, dictionary_learner, D, pool, output_directory = ""):
        print("write clusters")
        self.dictionary_learner = dictionary_learner
        self.chunk_size = self.dictionary_learner.chunk_size
        self.chunk_size = 2**18

        self.dictionary_learner.set_dictionary(D)


        clusters_mm = np.memmap(output_directory+ "/kmer_clusters", dtype='int16', mode='w+', shape=(5, len(self.non_zero_columns)), order='F')
        clusters = clusters_mm[0,:]


        iter_nb = int(len(self.non_zero_columns)/self.chunk_size) + 1
        print(iter_nb)
        di = np.arange(len(self.non_zero_columns), dtype = np.int32)[self.non_zero_columns]

        #r = pool.map(self.write_part, range(0, iter_nb))
        processes = []
        for i in range(iter_nb):
            p = Process(target=self.write_part, args= i)
            processes.append[p]

        [x.start() for x in processes]

        """
        for i in range(iter_nb):
            print(i)
            inds, c = r.next()
            clusters[di[inds]] = c[:]
            clusters.flush()
        """
