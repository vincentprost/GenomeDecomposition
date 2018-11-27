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
import math
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Optional app description')


parser.add_argument('-i', '--input',  type=str,
                    help='input folder')

parser.add_argument('-gw', '--global_weights',  type=str, default="../LatentStrainAnalysis-master_light/matrices/global_weights_el1_27",
                    help='global weights')

args = parser.parse_args()


cluster_cols = 5


s = os.path.getsize(args.input + '/kmer_clusters')
hash_size = int(np.log(s/(cluster_cols * 2)) /np.log(2))

print("hash_size : " + str(hash_size))



ncols = 2**hash_size
clusters = np.memmap(args.input + '/kmer_clusters', dtype='int16', mode='r', shape=(5, 2**hash_size), order='F')


nz = clusters[0,:] != 0
nzi = np.sum(nz)
print(nzi)

clusts = np.sort(np.unique(clusters[0,:]))
clusts = clusts[clusts != 0]
print(clusts)
print(len(clusts))


m = np.max(clusts)


try:
	GW = np.load(args.global_weights)
except:
	s = os.path.getsize(args.global_weights)
	GW = np.memmap(args.global_weights, dtype='float32', mode='r', shape=(2**hash_size))


CP = np.zeros(m)
CP2 = np.zeros(m)

global_weight_sum = GW[nz].sum()
print("global_weight_sum : " + str(global_weight_sum))
cluster_sizes = np.zeros(m, dtype=np.uint64)

print("compute probs and sizes")


def find_indexes(i):
	# Not optimal
	#print(np.where(clusters == i))
	print(i)
	c = np.where(clusters == i)[1]


	cp = GW[c].sum()/global_weight_sum
	cp2 = len(c)/ncols

	return cp, cp2



p = multiprocessing.Pool(3)
r = p.imap(find_indexes, clusts)

for i in clusts:
	cp, cp2 = r.next()
	CP[i - 1] += cp
	CP2[i - 1] +=  cp2

print(CP)
print(np.sum(CP))

np.save(args.input + '/cluster_probs', CP)
CP_mm = np.memmap(args.input + "/cluster_probs", mode = 'w+', dtype = np.float32, shape = np.shape(CP))
CP_mm[:] = CP[:]
CP_mm2 = np.memmap(args.input + "/cluster_probs_cp", mode = 'w+', dtype = np.float32, shape = np.shape(CP))
CP_mm2[:] = CP2[:]


if not os.path.exists(args.input + "/partitions"):
    os.makedirs(args.input + "/partitions")


del CP_mm
