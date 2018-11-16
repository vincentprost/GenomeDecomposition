#!/usr/bin/env python

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
#import pathos.multiprocessing as multiprocessing

import argparse
from dictionary_learner import DictionaryLearner, ClusterWriter, write_clusters

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

CW = ClusterWriter()
CW.read_data(args.input + "/abundance_el1_27_cwn", nrows = n, ncols = 2**hash_size,
    non_zero_columns = args.input + "/non_zeros_el1_27")


print(CW.vectors)
print(np.shape(CW.vectors))
p = multiprocessing.Pool(1)

DL = DictionaryLearner()
D = DL.learn_dictionary(CW.vectors)
print(D)

write_clusters(DL, CW.vectors, CW.non_zero_columns,  args.output)
#CW.write_clusters(DL, D, p, args.output)
