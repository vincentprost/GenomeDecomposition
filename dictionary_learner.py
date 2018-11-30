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
import multiprocessing as mp



def eval_dictionary(D, G):
    D = D /  np.sqrt(np.sum(D * D, 0))
    D = np.nan_to_num(D)
    dd = D.T @ G
    err = np.max(dd, 0)
    err3 = np.sum(2 - 2* err)
    return err3




class DictionaryLearner:

    def __init__(self, K = 200, cpu = 5, iter_nb = 500, chunk_size = 2000,
    sparse_coding_type = "omp", opt_function = "F", lambda1 = 0.1, lambda2 = 0.1, eps = 0.1):
        np.random.seed(0)
        self.K = K
        self.cpu = 5
        self.chunk_size = chunk_size
        self.iter_nb = iter_nb
        self.lasso_params = { 'mode' : 0,
                  'lambda1' : lambda1, 'lambda2' : lambda2, 'pos' : True, 'numThreads' : 2
        }

        self.omp_eps = 0.1
        self.sparse_coding_type = sparse_coding_type
        self.opt_function = opt_function
        self.D = None
        print(self.sparse_coding_type)


    def __str__(self):
        return "DictionaryLearner"

    def set_dictionary(self, D):
        self.D = D

    def sparse_coding_lasso(self, i, vectors):
        A = np.zeros((self.K, self.K))
        B = np.zeros((self.n, self.K))
        inds = np.arange(i, i + self.chunk_size)
        v = vectors[:, inds]
        a = spams.lasso(v, D = self.D,**self.lasso_params).toarray()
        for k in np.arange(len(inds)):
        	A += np.outer(a[:,k],a[:,k])
        	B += np.outer(v[:,k],a[:,k])
        return A, B

    def sparse_coding_omp(self, i, vectors):
        A = np.zeros((self.K, self.K))
        B = np.zeros((self.n, self.K))
        inds = np.arange(i, i + self.chunk_size)
        v = vectors[:, inds]
        a = spams.omp(v, D = self.D, eps = self.omp_eps, return_reg_path = False, numThreads = self.cpu).toarray()
        #a = spams.omp(v, D = self.D, L = 3, return_reg_path = False).toarray()
        for k in np.arange(len(inds)):
            A += np.outer(a[:,k],a[:,k])
            B += np.outer(v[:,k],a[:,k])
        return A, B


    def sparse_coding_very_sparse(self, i, vectors):
        A = np.zeros((self.K, self.K))
        B = np.zeros((self.n, self.K))

        inds = np.arange(i, i + self.chunk_size)
        v = vectors[:, inds]
        #clusts = np.zeros(len(inds), dtype = np.int32)

        for k in np.arange(len(inds)):
            d = v[:,k].dot(self.D)
            clust = np.nanargmax(d)
            alpha = np.dot(self.D[:, clust], v[:,k])/np.dot(self.D[:, clust], self.D[:, clust])
            A[clust, clust] += alpha ** 2
            B[:, clust] += alpha * v[:,k]
        return A, B

    def sparse_coding_very_sparse_kl(self, i, vectors):
        A = np.zeros((self.K, self.K))
        B = np.zeros((self.n, self.K))

        inds = np.arange(i, i + self.chunk_size)
        v = vectors[:, inds]
        lD = np.log(self.D)
        for k in np.arange(len(inds)):
            kld = v[:,k].dot(lD)
            clust = np.nanargmax(kld)
            alpha = np.sum(self.D[:, clust]) / np.sum(v[:,k])
            A[clust, clust] += alpha
            B[:, clust] += alpha * v[:,k]
        return A, B


    def sparse_coding(self, i, vectors):
        if self.sparse_coding_type == "lasso":
            return self.sparse_coding_lasso(i, vectors)
        elif self.sparse_coding_type == "omp":
            return self.sparse_coding_omp(i, vectors)
        elif self.sparse_coding_type == "very_sparse":
            if self.opt_function == "F":
                return self.sparse_coding_very_sparse(i, vectors)
            elif self.opt_function == "KL":
                return self.sparse_coding_very_sparse_kl(i, vectors)


    def update_dictionary(self, A, B):
        if self.opt_function == "F":
            return self.update_dictionary_F(A, B)
        elif self.opt_function == "KL":
            return self.update_dictionary_KL(A, B)


    def update_dictionary_F(self, A, B):
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
        #return D /  np.sqrt(np.sum(D * D, 0))
        return D


    def update_dictionary_KL(self, A, B):
        A[A == 0] = 0.1
        D = B/np.diagonal(A)
        D[D == np.inf] = 1
        return D


    def learn_dictionary(self, vectors):
        self.n = np.shape(vectors)[0]
        self.init_dictionary(vectors)
        A = np.zeros((self.K, self.K), dtype = np.float64)
        B = np.zeros((self.n, self.K), dtype = np.float64)

        for k in np.arange(2, self.iter_nb):
            offset = np.random.randint(np.shape(vectors)[1] - self.chunk_size - 1)
            beta = 0.7
            A_, B_ = self.sparse_coding(offset, vectors)
            A = beta * A + A_
            B = beta * B + B_
            D = self.update_dictionary(A, B)
            self.D = D
        return self.D



    def learn_dictionary_eval(self, vectors, G):

        self.n = np.shape(vectors)[0]
        self.init_dictionary(vectors)
        A = np.zeros((self.K, self.K), dtype = np.float64)
        B = np.zeros((self.n, self.K), dtype = np.float64)
        err = np.zeros(self.iter_nb)

        for k in np.arange(self.iter_nb):
            print("iter " + str(k))
            offset = np.random.randint(np.shape(vectors)[1] - self.chunk_size - 1)
            beta = 0.7
            A_, B_ = self.sparse_coding(offset, vectors)
            A = beta * A + A_
            B = beta * B + B_
            D = self.update_dictionary(A, B)
            self.D = D
            err[k] = eval_dictionary(D, G)
        return self.D, err


    def init_dictionary(self, vectors):

        self.D = np.asfortranarray(np.zeros((self.n, self.K), dtype = np.float32))
        for i in range(self.K):
        	ind = np.random.randint(np.shape(vectors)[1])
        	self.D[:, i] = vectors[:, ind]


    def find_best_clusters_F(self, vectors):
        dd = distance.cdist(vectors.T, self.D.T, 'cosine')
        clusts = np.nanargmin(dd, axis = 1)
        vals = dd[np.arange(len(clusts)), clusts]
        ol = vals > np.inf
        clusters = np.zeros((np.shape(vectors)[1],), dtype = np.int16)
        clusters[~ol] = clusts[~ol] + 1
        return clusters


    def find_best_clusters_kl(self, vectors):
        lD = np.log(self.D)
        clusters = np.zeros((np.shape(vectors)[1],), dtype = np.int16)
        for k in np.arange(np.shape(vectors)[1]):
            kld = vectors[:,k].dot(lD)
            clust = np.nanargmax(kld)
            clusters[k] = clust + 1
        return clusters


    def find_best_clusters(self, vectors):
        if self.opt_function == "F":
            return self.find_best_clusters_F(vectors)
        if self.opt_function == "KL":
            return self.find_best_clusters_F(vectors)





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

    def read_data(self, input_data,  nrows = 0, ncols = 0, data_type = 'float32', non_zero_columns = None, order='F'):

        error_message = "error, could not read data"
        matrix_loaded = False
        if non_zero_columns is not None:
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


    def get_vectors(self, vectors):
        return self.vectors

    def set_vectors(self, vectors):
        self.vectors = vectors


    def write_part(self, i):
        print("start ok ")

        sup = min(np.shape(self.vectors)[1], (i + 1) * self.chunk_size)
        inds = np.arange(i * self.chunk_size, sup)

        v = self.vectors[:, inds]

        clusters = self.dictionary_learner.find_best_clusters(i, v)
        return inds, clusters


    def write_clusters(self, dictionary_learner, D, output_directory = "", non_zero_columns = None, clusters_nb = 5):
        print("write clusters")
        if non_zero_columns is not None:
            self.non_zero_columns = non_zero_columns

        ncols = len(self.non_zero_columns)
        nzi = np.shape(self.vectors)[1]

        di = np.arange(ncols, dtype = np.int32)[self.non_zero_columns]

        chunk_size = 2**18

        # TODO set independance between D and write clusters

        dictionary_learner.set_dictionary(D)

        clusters_mm = np.memmap(output_directory+ "/kmer_clusters", dtype='int16', mode='w+', shape=(5, ncols), order='F')
        clusters = clusters_mm[0,:]

        iter_nb = int(nzi/chunk_size) + 1
        print(iter_nb)
        arguments = []
        for i in range(iter_nb + 1):
            sup = min(nzi, (i + 1) * chunk_size)
            arguments = arguments + [self.vectors[:, np.arange(i * chunk_size, sup)]]


        p = mp.Pool(4)
        r = p.imap(dictionary_learner.find_best_clusters, arguments)

        for i in range(iter_nb):
            print(i)
            sup = min(nzi, (i + 1) * chunk_size)
            inds = np.arange(i * chunk_size, sup)
            c = r.next()
            clusters[di[inds]] = c[:]
            clusters.flush()
