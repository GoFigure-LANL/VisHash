### ©2020. Triad National Security, LLC. All rights reserved.
### This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

import argparse
import numpy as np
import time

import util
import filehandling


def calc_distances(ftr):
    print(time.asctime(), ' Computing distances')
    start = time.time()
    dst = util.compute_pair_distances(ftr)
    end = time.time()
    print(time.asctime(), ' Done Computing distances in ', end-start, ' seconds', flush=True)

    return dst


def get_min_distances(dst, n_matches):
    print(time.asctime(), ' Sorting distances')
    start = time.time()
    midx = util.mindist(dst, n=n_matches)
    end = time.time()
    print(time.asctime(), ' Done Sorting distances in ', end-start, ' seconds', flush=True)
    return midx



###
### Main and commandline arguments
###
def get_args():
    parser = argparse.ArgumentParser(description='Calculate distances between all-pairs of given signature list.')
    parser.add_argument('--postfix', type=str,
                        help='filenames_[postfix].csv and signatures_[postfix].npy will be read while mindist_[postfix].csv will be written')
    parser.add_argument('--path', type=str, default='./',
                        help='Path to input files [./]')
    parser.add_argument('--n_matches', type=int, default=1000,
                        help='If specified limit number of matches n_matches (strongly recommended) [1000]')
    args = parser.parse_args()
    return(args)


def main():
    args = get_args()
    postfix = args.postfix

    # Read signatures and filename list
    filenames = filehandling.read_filenames(postfix, args.path)
    sigs = np.load(args.path + '/signatures_' + postfix + '.npy')
    
    # Compute all-pairs distances
    dist_mat = calc_distances(sigs)
    midx = get_min_distances(dist_mat, args.n_matches)
    filehandling.save_matchlist(midx, dist_mat, filenames, postfix)
    

if __name__ == '__main__':
    main()
